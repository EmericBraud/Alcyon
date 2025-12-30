#include "engine/engine.hpp"
//clang-format off
static constexpr int MVV_LVA_TABLE[7][7] = {
    // Attaquants:  P      N      B      R      Q      K     NONE
    /* PAWN   */ {105, 104, 103, 102, 101, 100, 0},
    /* KNIGHT */ {205, 204, 203, 202, 201, 200, 0},
    /* BISHOP */ {305, 304, 303, 302, 301, 300, 0},
    /* ROOK   */ {405, 404, 403, 402, 401, 400, 0},
    /* QUEEN  */ {505, 504, 503, 502, 501, 500, 0},
    /* KING   */ {605, 604, 603, 602, 601, 600, 0},
    /* NONE   */ {0, 0, 0, 0, 0, 0, 0}};
//clang-format on

void Engine::play(int time_ms)
{
    time_limit_ms = time_ms;
    time_up = false;
    start_time = Clock::now();

    Move best_move = 0;
    int last_score = 0;
    int window = 40;

    total_nodes = 0;
    tt_cuts = 0;
    beta_cutoffs = 0;
    clear_killers();
    age_history();

    for (int current_depth = 1; current_depth <= MAX_DEPTH; ++current_depth)
    {
        if (time_up)
            break;
        int delta = 16;
        int alpha = last_score - window;
        int beta = last_score + window;

        // Si on est à faible profondeur, on garde INF pour la stabilité
        if (current_depth < 3)
        {
            alpha = -INF;
            beta = INF;
        }

        // On n'active l'aspiration qu'à partir d'une certaine profondeur
        if (current_depth >= 5)
        {
            alpha = std::max(-INF, last_score - delta);
            beta = std::min(INF, last_score + delta);
        }

        while (true)
        {
            int score = negamax(current_depth, alpha, beta, 0);

            if (time_up)
                break;

            if (score <= alpha) // Fail Low
            {
                // On élargit la fenêtre vers le bas
                alpha = std::max(-INF, alpha - delta);
                beta = (alpha + beta) / 2; // On peut resserrer beta pour aider l'élagage
                delta += delta / 4 + 5;    // Croissance dynamique du delta
                std::cout << "Fail Low (new alpha: " << alpha << ")\n";
            }
            else if (score >= beta) // Fail High
            {
                // On élargit la fenêtre vers le haut
                beta = std::min(INF, beta + delta);
                delta += delta / 4 + 5;
                std::cout << "Fail High (new beta: " << beta << ")\n";
            }
            else
            {
                // SUCCESS : Le score est dans la fenêtre
                last_score = score;
                break;
            }

            // Sécurité : si la recherche prend trop de temps ou si delta explose
            if (delta > 1000)
            {
                alpha = -INF;
                beta = INF;
            }
        }

        if (!time_up)
        {
            best_move = tt.get_move(board.get_hash());
            std::cout << "Depth " << current_depth
                      << " | Score " << last_score
                      << " | Nodes " << total_nodes
                      << " | PV: " << get_pv_line(current_depth)
                      << std::endl;
        }
        if (std::abs(last_score) >= MATE_SCORE - MAX_DEPTH) // We found a checkmate
        {
            break;
        }
    }

    if (best_move.get_value() != 0)
        board.play(best_move);
}

int Engine::score_move(const Move &move, const Board &board, const Move &tt_move, int ply, const Move &prev_move) const
{
    // 1. Mise en cache de la valeur brute du mouvement (32 bits)
    const uint32_t move_val = move.get_value();
    if (move_val == tt_move.get_value())
        return 3000000;

    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();

    // 2. Captures et Promotions
    if (to_piece != NO_PIECE || flags == Move::Flags::EN_PASSANT_CAP || move.is_promotion())
    {
        const Piece target = (flags == Move::Flags::EN_PASSANT_CAP) ? PAWN : to_piece;
        const int mvv_lva = MVV_LVA_TABLE[target][from_piece];

        if (move.is_promotion())
            return 2000000 + mvv_lva;

        // On évite SEE si la capture est manifestement "bonne" (Victime >= Attaquant)
        if (Eval::get_piece_score(target) >= Eval::get_piece_score(from_piece))
            return 1000000 + mvv_lva;

        int see_value = see(move.get_to_sq(), target, from_piece, board.get_side_to_move(), move.get_from_sq());

        // SEE est appelé ici (5.06% du temps total)
        if (see_value == 0)
            return 950000 + MVV_LVA_TABLE[target][from_piece];

        return 100000 + mvv_lva;
    }

    // 3. Coups calmes
    if (move_val == killer_moves[ply][0].get_value())
        return 900000;
    if (move_val == killer_moves[ply][1].get_value())
        return 800000;

    const Color us = board.get_side_to_move();
    // prev_move est maintenant passé en argument (extraite une seule fois du history.back())
    if (ply > 0 && prev_move != 0 && move_val == counter_moves[us][prev_move.get_from_piece()][prev_move.get_to_sq()].get_value())
        return 700000;

    return history_moves[us][move.get_from_sq()][move.get_to_sq()];
}

std::string Engine::get_pv_line(int depth)
{
    std::string pv_line = "";
    std::vector<Move> moves_to_unplay;

    for (int i = 0; i < depth; i++)
    {
        Move m = tt.get_move(board.get_hash());
        if (m == 0)
            break;

        pv_line += m.to_uci() + " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

    // On restaure l'état original sans aucune copie
    for (int i = (int)moves_to_unplay.size() - 1; i >= 0; i--)
    {
        board.unplay(moves_to_unplay[i]);
    }
    return pv_line;
}

int Engine::negamax(int depth, int alpha, int beta, int ply)
{
    // 1. Détection des nullités (Répétition / 50 coups)
    if (ply > 0)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
        {
            return (board.get_history_size() < 20) ? -25 : 0;
        }
    }

    // 2. Sondage de la Transposition Table (TT)
    int tt_score;
    Move tt_move = 0;
    total_nodes++;
    bool tt_hit = tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move);

    if (tt_hit && ply > 0)
    {
        tt_cuts++;
        return tt_score;
    }

    if ((total_nodes & 2047) == 0)
        check_time();
    if (time_up)
    {

        return Eval::eval_relative(board.get_side_to_move(), board, alpha, beta);
    }

    // 3. Quiescence Search à l'horizon
    if (depth <= 0)
        return qsearch(alpha, beta);

    // 4. Null Move Pruning (NMP)
    bool in_check = board.is_king_attacked((Color)board.get_side_to_move());
    if (depth >= 3 && ply > 0 && !in_check)
    {
        int stored_ep;
        board.play_null_move(stored_ep);
        int R = (depth > 6) ? 3 : 2;
        int score = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1);
        board.unplay_null_move(stored_ep);

        if (score >= beta)
        {
            if (score >= MATE_SCORE - MAX_DEPTH)
                return beta;
            return score;
        }
    }
    // --- Essayer le TT Move AVANT la génération ---
    int best_score = -INF;
    Move best_move_this_node = 0;
    int moves_searched = 0;
    const Color player = board.get_side_to_move();

    if (tt_move.get_value() != 0)
    {
        const U64 next_hash = board.get_hash_after(tt_move);
        tt.prefetch(next_hash);
        board.play(tt_move);
        if (!board.is_king_attacked((Color)player))
        {
            moves_searched++;
            // Premier coup : fenêtre complète
            int score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            if (ply > 0)
                _mm_prefetch((const char *)&(board.get_history()->back()), _MM_HINT_T0);
            board.unplay(tt_move);

            if (score >= beta)
                return score; // Beta-cutoff immédiat

            if (score > best_score)
            {
                best_score = score;
                best_move_this_node = tt_move;
                if (score > alpha)
                    alpha = score; // MISE À JOUR D'ALPHA
            }
        }
        else
        {
            if (ply > 1)
                _mm_prefetch((const char *)&(board.get_history())[board.get_history()->size() - 2], _MM_HINT_T0);
            if (ply > 0)
                _mm_prefetch((const char *)&(board.get_history()->back()), _MM_HINT_T0);
            board.unplay(tt_move);
        }
    }

    const Move m = ply > 0 ? (board.get_history()->back()).move : 0;

    // 5. Génération et tri des coups
    MoveList list;
    MoveGen::generate_pseudo_legal_moves(board, board.get_side_to_move(), list);

    for (int i = 0; i < list.count; ++i)
        list.scores[i] = score_move(list.moves[i], board, tt_move, ply, m);

    int alpha_orig = alpha;

    // --- BOUCLE DE RECHERCHE PVS ---
    for (int i = 0; i < list.count; ++i)
    {
        if ((total_nodes & 2047) == 0)
            check_time();
        if (time_up)
            break;

        Move &m = list.pick_best_move(i);
        if (m.get_value() == tt_move.get_value())
            continue;
        if (!board.is_move_legal(m))
            continue;
        int score;
        const int m_score = list.scores[i];
        bool is_tactical = (m_score >= 900000);

        const U64 next_hash = board.get_hash_after(m);
        tt.prefetch(next_hash);
        board.play(m);
        moves_searched++;

        if (moves_searched == 1)
        {
            // PREMIER COUP : Recherche avec fenêtre complète
            score = -negamax(depth - 1, -beta, -alpha, ply + 1);
        }
        else
        {
            int reduction = 0;

            // Conditions pour réduire : profondeur >= 3 et coup "calme" (non tactique)
            if (depth >= 3 && moves_searched > 4 && !is_tactical && !in_check)
            {
                // Accès direct à la table (plus rapide que les calculs de log)
                reduction = static_cast<int>(lmr_table[std::min(depth, 63)][std::min(moves_searched, 63)]);

                // Sécurité : on ne réduit pas au point de tomber en qsearch immédiatement
                reduction = std::clamp(reduction, 0, depth - 2);
            }

            // Recherche réduite avec fenêtre nulle
            score = -negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);

            // Re-search si le coup réduit dépasse alpha (il est peut-être meilleur que prévu)
            if (score > alpha && reduction > 0)
            {
                score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
            }

            // Re-search complet (PVS) si le coup est dans la fenêtre
            if (score > alpha && score < beta)
            {
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            }
        }
        if (ply > 0)
            _mm_prefetch((const char *)&(board.get_history()->back()), _MM_HINT_T0);

        board.unplay(m);

        if (score >= beta)
        {
            beta_cutoffs++;
            if (ply > 0)
                _mm_prefetch((const char *)&(board.get_history()->back()), _MM_HINT_T0);
            tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {
                if (ply > 0)
                {
                    Move prev_move = (board.get_history()->back()).move;
                    if (prev_move.get_value() != 0)
                    {
                        counter_moves[player][prev_move.get_from_piece()][prev_move.get_to_sq()] = m;
                    }
                }
                history_moves[player][m.get_from_sq()][m.get_to_sq()] += depth * depth;
                if (!(m.get_value() == killer_moves[ply][0].get_value()))
                {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = m;
                }
            }
            return score;
        }

        if (score > best_score)
        {
            best_score = score;
            best_move_this_node = m;
            if (score > alpha)
                alpha = score;
        }
    }
    if (moves_searched == 0)
    {
        // On vérifie si le camp qui doit jouer est en échec
        if (board.is_king_attacked(board.get_side_to_move()))
            return -MATE_SCORE + ply; // MAT
        else
            return 0; // PAT
    }

    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

    return best_score;
}