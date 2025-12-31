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

int SearchWorker::score_move(const Move &move, const Board &board, const Move &tt_move, int ply, const Move &prev_move) const
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

std::string SearchWorker::get_pv_line(int depth)
{
    std::string pv_line = "";
    std::vector<Move> moves_to_unplay;

    for (int i = 0; i < depth; i++)
    {
        Move m = shared_tt.get_move(board.get_hash());
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

int SearchWorker::negamax_with_aspiration(int depth, int last_score)
{
    int delta = 16;
    int alpha = -INF;
    int beta = INF;

    // On n'active l'aspiration qu'après une profondeur minimale
    if (depth >= 5)
    {
        alpha = std::max(-INF, last_score - delta);
        beta = std::min(INF, last_score + delta);
    }

    while (true)
    {
        int score = negamax(depth, alpha, beta, 0);

        // Si le temps est écoulé ou qu'un autre thread a demandé l'arrêt
        if (shared_stop.load(std::memory_order_relaxed))
            return score;

        if (score <= alpha)
        { // Fail Low
            alpha = std::max(-INF, alpha - delta);
            beta = (alpha + beta) / 2; // Resserrage de beta pour aider l'élagage
            delta += delta / 4 + 5;
        }
        else if (score >= beta)
        { // Fail High
            beta = std::min(INF, beta + delta);
            delta += delta / 4 + 5;
        }
        else
        {
            return score; // Succès : score dans la fenêtre
        }

        // Sécurité pour éviter les boucles infinies hors limites
        if (delta > 2000)
        {
            alpha = -INF;
            beta = INF;
        }
    }
}
int SearchWorker::negamax(int depth, int alpha, int beta, int ply)
{
    // 1. Vérification périodique de l'arrêt (Atomique)
    // On incrémente le compteur global de nœuds et on vérifie le temps
    if (((++local_nodes)) & 2047 == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;
        if (thread_id == 0)
        {
            auto now = Clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_ref).count();
            if (elapsed >= time_limit_ms_ref)
            {
                shared_stop.store(true, std::memory_order_relaxed);
            }
        }
        // Si l'arrêt est demandé (par temps ou par un autre thread), on quitte
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;
    }

    // 2. Détection des nullités (Répétition / 50 coups)
    if (ply > 0)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            return (board.get_history_size() < 20) ? -25 : 0;
    }

    // 3. Sondage de la Transposition Table (TT)
    int tt_score;
    Move tt_move = 0;
    bool tt_hit = shared_tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move);

    if (tt_hit && ply > 0)
        return tt_score;

    // 4. Quiescence Search à l'horizon
    if (depth <= 0)
        return qsearch(alpha, beta);

    // 5. Null Move Pruning (NMP)
    bool in_check = board.is_king_attacked(board.get_side_to_move());
    if (depth >= 3 && ply > 0 && !in_check)
    {
        int stored_ep;
        board.play_null_move(stored_ep);
        int R = (depth > 6) ? 3 : 2;
        int score = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1);
        board.unplay_null_move(stored_ep);

        if (score >= beta)
            return (score >= MATE_SCORE - MAX_DEPTH) ? beta : score;
    }

    // --- RECHERCHE DU TT MOVE (PV Move) ---
    int best_score = -INF;
    Move best_move_this_node = 0;
    int moves_searched = 0;
    const Color player = board.get_side_to_move();

    if (tt_move.get_value() != 0)
    {
        board.play(tt_move);
        if (!board.is_king_attacked(player))
        {
            moves_searched++;
            best_score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            board.unplay(tt_move);

            if (best_score >= beta)
                return best_score;
            if (best_score > alpha)
                alpha = best_score;
            best_move_this_node = tt_move;
        }
        else
        {
            board.unplay(tt_move);
        }
    }

    // 6. Génération et tri des coups restants
    MoveList list;
    MoveGen::generate_pseudo_legal_moves(board, player, list);

    // Récupération du coup précédent pour les Counter-moves
    const Move prev_m = ply > 0 ? (board.get_history()->back()).move : 0;

    for (int i = 0; i < list.count; ++i)
        list.scores[i] = score_move(list.moves[i], board, tt_move, ply, prev_m);

    int alpha_orig = alpha;

    // 7. BOUCLE PVS (Principal Variation Search)
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);

        if (m.get_value() == tt_move.get_value())
            continue;
        if (!board.is_move_legal(m))
            continue;

        moves_searched++;
        int score;
        bool is_tactical = (list.scores[i] >= 900000);

        board.play(m);

        // --- LATE MOVE REDUCTION (LMR) ---
        if (depth >= 3 && moves_searched > 4 && !is_tactical && !in_check)
        {
            int r = static_cast<int>(lmr_table[std::min(depth, 63)][std::min(moves_searched, 63)]);
            r = std::clamp(r, 0, depth - 2);

            score = -negamax(depth - 1 - r, -alpha - 1, -alpha, ply + 1);

            // Re-search si le coup réduit semble bon
            if (score > alpha)
                score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
        }
        else if (moves_searched > 1) // Null Window Search pour PVS
        {
            score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
        }
        else // Full Window Search (seulement si moves_searched == 1 et pas de TT move)
        {
            score = -negamax(depth - 1, -beta, -alpha, ply + 1);
        }

        // Si le score est dans la fenêtre mais pas une coupure, on re-cherche normalement
        if (score > alpha && score < beta && moves_searched > 1)
        {
            score = -negamax(depth - 1, -beta, -alpha, ply + 1);
        }

        board.unplay(m);

        // Vérification d'arrêt immédiat après un unplay
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;

        // --- MISE À JOUR DES SCORES ET DES TABLES ---
        if (score >= beta)
        {
            shared_tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {
                // Update Counter-moves et History (Local au thread)
                if (prev_m.get_value() != 0)
                    counter_moves[player][prev_m.get_from_piece()][prev_m.get_to_sq()] = m;

                history_moves[player][m.get_from_sq()][m.get_to_sq()] += depth * depth;

                if (m.get_value() != killer_moves[ply][0].get_value())
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

    // 8. Gestion des Mats et Pats
    if (moves_searched == 0)
    {
        return in_check ? -MATE_SCORE + ply : 0;
    }

    // 9. Sauvegarde TT Finale
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

    return best_score;
}