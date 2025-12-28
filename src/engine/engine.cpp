#include "engine/engine.hpp"

void Engine::play(int time_ms)
{
    time_limit_ms = time_ms;
    time_up = false;
    start_time = Clock::now();

    Move best_move;
    int last_score = 0; // On commence à 0
    int window = 30;    // Taille de la fenêtre (environ 1/3 de pion)

    total_nodes = 0;
    tt_cuts = 0;
    beta_cutoffs = 0;
    clear_killers();
    age_history();

    for (int current_depth = 1; current_depth <= MAX_DEPTH; ++current_depth)
    {
        if (time_up)
            break;

        int alpha = last_score - window;
        int beta = last_score + window;

        // Si on est à faible profondeur, on garde INF pour la stabilité
        if (current_depth < 3)
        {
            alpha = -INF;
            beta = INF;
        }

        while (true)
        {
            int score = negamax(current_depth, alpha, beta, 0);

            if (time_up)
                break;

            if (score <= alpha)
            {
                // FAIL LOW : On est moins bon que prévu, on descend alpha
                alpha -= window;
                window *= 2; // On élargit pour la prochaine fois
            }
            else if (score >= beta)
            {
                // FAIL HIGH : On est meilleur que prévu, on monte beta
                beta += window;
                window *= 2;
            }
            else
            {
                // SUCCESS : Le score est dans la fenêtre
                last_score = score;
                // On réduit un peu la fenêtre pour la profondeur suivante
                window = 30 + std::abs(last_score) / 10;
                break;
            }

            // Sécurité : si la fenêtre devient trop large, on passe en INF
            if (alpha < -900000)
                alpha = -INF;
            if (beta > 900000)
                beta = INF;
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

int Engine::score_move(const Move &move, const Board &board, const Move &tt_move, int ply) const
{
    if (tt_move.get_value() != 0 && move.get_value() == tt_move.get_value())
        return 2000000;

    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();

    // Priorité aux promotions
    if (flags == Move::Flags::PROMOTION_MASK)
        return 1500000;

    // Gestion des captures avec SEE
    if (to_piece != NO_PIECE || flags == Move::Flags::EN_PASSANT_CAP)
    {
        Piece target = (flags == Move::Flags::EN_PASSANT_CAP) ? PAWN : to_piece;

        // Calcul de la SEE
        int see_value = see(move.get_to_sq(), target, from_piece, board.get_side_to_move(), move.get_from_sq());

        if (see_value >= 0)
        {
            // Capture gagnante ou égale : on garde le MVV-LVA
            return 1000000 + (eval::get_piece_score(target) * 10 - eval::get_piece_score(from_piece));
        }
        else
        {
            // Capture perdante : on la place APRÈS les coups calmes (killers/history)
            return 100000 + see_value;
        }
    }

    // Coups calmes (Killers et History)
    if (move.get_value() == killer_moves[ply][0].get_value())
        return 900000;
    if (move.get_value() == killer_moves[ply][1].get_value())
        return 800000;

    return history_moves[board.get_side_to_move()][move.get_from_sq()][move.get_to_sq()];
}

std::string Engine::get_pv_line(int depth)
{
    std::string pv_line = "";
    Board temp_board = board; // Copie locale pour ne pas corrompre le plateau réel

    for (int i = 0; i < depth; i++)
    {
        Move m = tt.get_move(temp_board.get_hash());
        if (m.get_value() == 0)
            break; // Plus de coup en cache

        pv_line += m.to_uci() + " ";
        temp_board.play(m);
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
    Move tt_move;
    total_nodes++;
    bool tt_hit = tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move);

    if (tt_hit && ply > 0)
    {
        tt_cuts++;
        return tt_score;
    }

    check_time();
    if (time_up)
        return eval::eval_relative(board.get_side_to_move(), board);

    // 3. Quiescence Search à l'horizon
    if (depth <= 0)
        return qsearch(alpha, beta);

    // 4. Null Move Pruning (NMP)
    bool in_check = MoveGen::is_king_attacked(board, board.get_side_to_move());
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
    Move best_move_this_node;
    int moves_searched = 0;
    const Color player = board.get_side_to_move();

    if (tt_move.get_value() != 0)
    {
        board.play(tt_move);
        if (!MoveGen::is_king_attacked(board, player))
        {
            moves_searched++;
            // Premier coup : fenêtre complète
            int score = -negamax(depth - 1, -beta, -alpha, ply + 1);
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
            board.unplay(tt_move);
        }
    }

    // 5. Génération et tri des coups
    MoveList list;
    MoveGen::generate_legal_moves(board, list);

    if (list.count == 0) // Pat ou Mat
    {
        return in_check ? (-MATE_SCORE + ply) : 0;
    }

    for (int i = 0; i < list.count; ++i)
        list.scores[i] = score_move(list.moves[i], board, tt_move, ply);

    int alpha_orig = alpha;

    // --- BOUCLE DE RECHERCHE PVS ---
    for (int i = 0; i < list.count; ++i)
    {
        check_time();
        if (time_up)
            break;

        Move &m = list.pick_best_move(i);
        if (m.get_value() == tt_move.get_value())
            continue;
        int score;
        const int m_score = list.scores[i];
        bool is_tactical = (m_score >= 900000);

        board.play(m);
        moves_searched++;

        if (moves_searched == 1)
        {
            // PREMIER COUP : Recherche avec fenêtre complète
            score = -negamax(depth - 1, -beta, -alpha, ply + 1);
        }
        else
        {
            // COUPS SUIVANTS : On parie qu'ils sont moins bons (Zero Window Search)

            // Tentative de réduction LMR
            int reduction = 0;
            if (depth >= 3 && moves_searched > 4 && !is_tactical && !in_check)
            {
                reduction = (depth > 6) ? 2 : 1;
            }

            // Recherche avec fenêtre nulle [-(alpha + 1), -alpha]
            score = -negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);

            // Re-search si le LMR a échoué (le score est meilleur que alpha)
            if (score > alpha && reduction > 0)
            {
                score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1);
            }

            // Re-search complet si le coup est réellement meilleur que notre PV actuelle
            if (score > alpha && score < beta)
            {
                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            }
        }

        board.unplay(m);

        if (score >= beta)
        {
            beta_cutoffs++;
            tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {
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

    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

    return best_score;
}