#include "engine/engine.hpp"
#include "engine/engine_manager.hpp"
//clang-format off
static constexpr int MVV_LVA_TABLE[7][7] = {
    // Attaquants:  P   N    B    R    Q    K  NONE
    /* PAWN   */ {105, 104, 103, 102, 101, 100, 0},
    /* KNIGHT */ {205, 204, 203, 202, 201, 200, 0},
    /* BISHOP */ {305, 304, 303, 302, 301, 300, 0},
    /* ROOK   */ {405, 404, 403, 402, 401, 400, 0},
    /* QUEEN  */ {505, 504, 503, 502, 501, 500, 0},
    /* KING   */ {605, 604, 603, 602, 601, 600, 0},
    /* NONE   */ {0, 0, 0, 0, 0, 0, 0}};
//clang-format on

static constexpr int TACTICAL_SCORE = 7000;

int SearchWorker::score_move(const Move &move, const Board &board, const Move &tt_move, int ply, const Move &prev_move) const
{
    const uint32_t move_val = move.get_value();
    if (move_val == tt_move.get_value())
        return 9600;

    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();

    // 1. Tactiques (Captures & Promotions)
    if (to_piece != NO_PIECE || flags == Move::Flags::EN_PASSANT_CAP || move.is_promotion())
    {
        const Piece target = (flags == Move::Flags::EN_PASSANT_CAP) ? PAWN : to_piece;
        const int mvv_lva = MVV_LVA_TABLE[target][from_piece];

        if (move.is_promotion())
        {
            // Priorise la promotion Dame
            const int promo_piece = move.get_promo_piece();
            if (promo_piece == QUEEN)
                return 9300;
            if (promo_piece == KNIGHT)
                return 8050;
            if (promo_piece == ROOK)
                return 8000;
            if (promo_piece == BISHOP)
                return 8000;
        }

        // On n'appelle SEE que si c'est potentiellement perdant (LVA prend MVV)
        if (Eval::get_piece_score(from_piece) > Eval::get_piece_score(target))
        {
            if (see(move.get_to_sq(), target, from_piece, board.get_side_to_move(), move.get_from_sq()) < 0)
                return -1000 + mvv_lva; // Clairement perdant
        }

        return 8600 + mvv_lva; // Captures normales/gagnantes
    }

    // 2. Coups calmes prioritaires
    if (move_val == killer_moves[ply][0].get_value())
        return 8000;
    if (move_val == killer_moves[ply][1].get_value())
        return 8000;

    const Color us = board.get_side_to_move();

    // Counter-move
    if (ply > 0 && prev_move != 0 && move_val == counter_moves[us][prev_move.get_from_piece()][prev_move.get_to_sq()].get_value())
        return 7500;

    // Bonus spécial pour les échecs "calmes" (Crucial pour mat en 11)
    // Attention : nécessite que ta MoveGen ou une fonction légère détecte l'échec
    // if (gives_check(move)) return 6000000;

    // 3. History Moves (Score relatif)
    return history_moves[us][move.get_from_sq()][move.get_to_sq()];
}
std::string SearchWorker::get_pv_line(int depth)
{
    std::string pv_line = "";
    std::vector<Move> moves_to_unplay;
    std::vector<uint64_t> visited_hashes;

    for (int i = 0; i < depth; i++)
    {
        Move m = shared_tt.get_move(board.get_hash());

        // 1. Si le coup est 0 (Nœud terminal, Mat, ou pas d'entrée), on arrête.
        if (m.get_value() == 0)
            break;

        // 2. Vérification de pseudo-légalité AVANT de toucher au board
        // Cela évite les asserts ou crashs dans is_move_legal si m est corrompu
        if (!board.is_move_pseudo_legal(m))
            break;

        if (!board.is_move_legal(m))
            break;

        // Détection de cycle
        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (uint64_t v : visited_hashes)
            if (v == h)
            {
                cycle_detected = true;
                break;
            }
        if (cycle_detected)
            break;

        visited_hashes.push_back(h);

        pv_line += m.to_uci() + " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

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

        if (depth >= 12 && std::abs(score) >= MATE_SCORE - 100)
        {
            return score;
        }

        if (shared_stop.load(std::memory_order_relaxed))
            return score;

        if (manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
            return score;
        }

        if (score <= alpha)
        {
            alpha = std::max(-INF, alpha - delta);
            delta += delta / 4 + 5;
            if (thread_id == 0)
                logs::debug << "info string fail low" << std::endl;
            ;
        }
        else if (score >= beta)
        {
            beta = std::min(INF, beta + delta);
            delta += delta / 4 + 5;
            if (thread_id == 0)
                logs::debug << "info string fail high" << std::endl;
        }
        else
        {
            if (!shared_stop.load(std::memory_order_relaxed))
                this->best_root_move = this->out_move;
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
void SearchWorker::iterative_deepening()
{
    int last_score = 0;
    for (int depth = 1; depth < MAX_DEPTH; ++depth)
    {
        last_score = negamax_with_aspiration(depth, last_score);
        if (shared_stop.load(std::memory_order_relaxed))
            return;
        if (thread_id == 0)
        {
            auto elapsed_ms = std::max<long long>(1,
                                                  std::chrono::duration_cast<std::chrono::milliseconds>(
                                                      std::chrono::steady_clock::now() - start_time_ref)
                                                      .count());

            long long nodes = global_nodes.load(std::memory_order_relaxed);
            long long nps = nodes * 1000 / elapsed_ms;
            logs::uci
                << "info depth " << depth
                << " score cp " << last_score
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << shared_tt.get_hashfull()
                << " pv " << get_pv_line(depth)
                << std::endl;
        }
    }
}
template <Color Us>
int SearchWorker::negamax(int depth, int alpha, int beta, int ply, bool allow_null)
{
    // 1. Vérification périodique de l'arrêt (Atomique)
    // On incrémente le compteur global de nœuds et on vérifie le temps
    if ((local_nodes & 32767) == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;

        if (thread_id == 0 && manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
        }
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;
    }
    ++local_nodes;

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

    bool in_check;
    bool in_check_tested = false;

    // 4. Quiescence Search à l'horizon
    if (depth <= 0)
    {
        in_check = board.is_king_attacked<Us>();
        in_check_tested = true;
        if (in_check && ply < MAX_DEPTH - 5)
        {
            depth = 1;
        }
        else
        {
            return qsearch<Us>(alpha, beta, ply);
        }
    }

    int best_score = -INF;
    Move best_move_this_node = 0;
    int moves_searched = 0;

    if (!in_check_tested)
        in_check = board.is_king_attacked<Us>();
    if (depth <= 7 && !in_check && ply > 0)
    {
        int static_eval = Eval::lazy_eval_relative<Us>(board);
        int margin = 120 * depth;
        if (static_eval - margin >= beta)
            return beta;
    }

    // --- Internal Iterative Deepening (IID) ---
    // Si on est dans un noeud PV, qu'on a de la profondeur, mais pas de coup TT
    if (tt_move == 0 && depth >= 6 && (beta - alpha > 1)) // Condition PV Node approximative
    {
        // On fait une recherche moins profonde pour trouver le meilleur coup
        int new_depth = depth - 4;

        // Appel récursif (note: on ne retourne pas le score, on veut juste remplir la TT)
        negamax<Us>(new_depth, alpha, beta, ply, true);

        // On récupère le coup que la recherche vient de trouver
        tt_move = shared_tt.get_move(board.get_hash());
    }

    // 5. Null Move Pruning (NMP)
    if (depth >= 3 && ply > 0 && allow_null && !in_check && beta < 9000 && alpha > -9000)
    {
        int stored_ep;
        board.play_null_move(stored_ep);
        int R = (depth > 6) ? 3 : 2;
        int score = -negamax<!Us>(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
        board.unplay_null_move(stored_ep);

        if (score >= beta)
            return (score >= MATE_SCORE - MAX_DEPTH) ? beta : score;
    }

    bool futil_pruning = false;
    int futil_margin = 0;

    if (depth <= 3 && !in_check && ply > 0 && (best_score > -MATE_SCORE + 100))
    {
        // Calcul de la marge (ajustable selon ton évaluation)
        futil_margin = 175 + 110 * depth;
        int static_eval = Eval::lazy_eval_relative<Us>(board);

        if (static_eval + futil_margin <= alpha)
        {
            futil_pruning = true;
        }
    }

    // 6. Génération et tri des coups restants
    MoveList list;
    MoveGen::generate_pseudo_legal_moves<Us>(board, list);

    // Récupération du coup précédent pour les Counter-moves
    const Move prev_m = ply > 0 ? (board.get_history()->back()).move : 0;

    uint64_t posKey = board.get_hash() ^ (uint64_t(thread_id) << 32);

    if (thread_id != 0 && ply > 5)
        for (int i = 0; i < list.count; ++i)
        {
            int base = score_move(list.moves[i], board, tt_move, ply, prev_m);

            uint64_t moveKey = (uint64_t)list.moves[i].get_value() * 0x9e3779b97f4a7c15ULL;
            uint64_t h = posKey ^ moveKey;

            uint64_t r = splitmix64(h);

            int delta = std::max(1, std::abs(base) / 12);
            int noise = int(r & 1023) - 512; // [-512 .. 511]
            noise = noise * delta / 512;

            list.scores[i] = base + noise;
        }
    else
        for (int i = 0; i < list.count; ++i)
            list.scores[i] = score_move(list.moves[i], board, tt_move, ply, prev_m);

    int alpha_orig = alpha;

    // 7. BOUCLE PVS (Principal Variation Search)
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);

        if (!board.is_move_legal(m))
            continue;

        int score;
        bool is_tactical = (list.scores[i] >= TACTICAL_SCORE);

        // --- LATE MOVE PRUNING (LMP) ---
        // Si on n'est pas en échec, à faible profondeur, on limite le nombre de coups calmes
        if (!in_check && depth <= 4 && !is_tactical)
        {
            // Formule classique : on teste de plus en plus de coups avec la profondeur
            int max_moves = 8 + (depth * depth * 2);
            if (moves_searched >= max_moves)
            {
                continue; // On ignore les coups calmes restants
            }
        }

        if (futil_pruning && moves_searched >= 1 && !is_tactical)
        {
            // On ne l'incrémente pas car on ne le cherche pas
            continue;
        }
        ++moves_searched;

        board.play<Us>(m);

        // --- LATE MOVE REDUCTION (LMR) ---
        if (depth >= 3 && moves_searched > 4 && !is_tactical && !in_check)
        {
            int r = static_cast<int>(lmr_table[std::min(depth, 63)][std::min(moves_searched, 63)]);
            r = std::clamp(r, 0, depth - 2);

            score = -negamax<!Us>(depth - 1 - r, -alpha - 1, -alpha, ply + 1, true);

            // Re-search si le coup réduit semble bon
            if (score > alpha)
                score = -negamax<!Us>(depth - 1, -alpha - 1, -alpha, ply + 1, true);
        }
        else if (moves_searched > 1) // Null Window Search pour PVS
        {
            score = -negamax<!Us>(depth - 1, -alpha - 1, -alpha, ply + 1, true);
        }
        else // Full Window Search (seulement si moves_searched == 1 et pas de TT move)
        {
            score = -negamax<!Us>(depth - 1, -beta, -alpha, ply + 1, true);
        }

        // Si le score est dans la fenêtre mais pas une coupure, on re-cherche normalement
        if (score > alpha && score < beta && moves_searched > 1)
        {
            score = -negamax<!Us>(depth - 1, -beta, -alpha, ply + 1, true);
        }

        board.unplay<Us>(m);

        // --- MISE À JOUR DES SCORES ET DES TABLES ---
        if (score >= beta)
        {
            shared_tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {
                int bonus = depth * depth;

                // On récompense le coup gagnant
                history_moves[Us][m.get_from_sq()][m.get_to_sq()] += bonus;

                // MALUS : On punit tous les coups calmes testés AVANT et qui ont échoué
                for (int j = 0; j < i; ++j)
                {
                    Move failed_move = list.moves[j];
                    // On ne punit que les coups calmes (pas les captures/promotions)
                    if (list.scores[j] < 8500)
                    {
                        history_moves[Us][failed_move.get_from_sq()][failed_move.get_to_sq()] = std::max(history_moves[Us][failed_move.get_from_sq()][failed_move.get_to_sq()] - bonus, -10000);
                    }
                }

                // 4. Update Killer Moves
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
            {
                alpha = score;
            }
        }
    }

    if (ply == 0 && !shared_stop.load(std::memory_order_relaxed))
    {

        this->out_move = best_move_this_node;
    }

    // 8. Gestion des Mats et Pats
    if (moves_searched == 0)
    {
        int score = in_check ? -MATE_SCORE + ply : 0;
        shared_tt.store(board.get_hash(), depth, ply, score, TT_EXACT, 0);
        return score;
    }

    // 9. Sauvegarde TT Finale
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

    return best_score;
}

template int SearchWorker::negamax<WHITE>(int depth, int alpha, int beta, int ply, bool allow_null);
template int SearchWorker::negamax<BLACK>(int depth, int alpha, int beta, int ply, bool allow_null);