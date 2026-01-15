#include "worker.hpp"

#include "engine/utils/random.hpp"
#include "engine/engine_manager.hpp"

namespace search
{
    inline bool is_null(const VBoard &board, int ply)
    {
        if (ply == 0)
            return false;
        return board.is_repetition() || board.get_halfmove_clock() >= 100;
    }

    template <Color Us>
    inline bool razoring(const VBoard &board, int depth, int alpha, bool is_pv, bool in_check)
    {
        if (in_check || is_pv || depth > 3)
            return false;

        int static_eval = Eval::lazy_eval_relative<Us>(board);
        int margin = 100 * depth;
        return static_eval + margin <= alpha;
    }

    inline bool should_qsearch(int &depth, int ply, bool in_check)
    {
        if (depth > 0)
            return false;
        if (in_check && ply < engine::config::search::MaxDepth - 5)
        {
            depth = 1;
            return false;
        }
        return true;
    }

    template <Color Us>
    inline bool reverse_futility_pruning(const VBoard &board, int depth, int ply, bool in_check, bool is_pv, int beta)
    {
        if (depth <= 7 && !in_check && ply > 0 && !is_pv)
        {
            int static_eval = Eval::lazy_eval_relative<Us>(board);
            int margin = 120 * depth;
            if (static_eval - margin >= beta)
                return true;
        }
        return false;
    }

    inline bool should_use_tt(bool tt_hit, int ply, bool is_pv, TTFlag flag, int tt_score, int beta)
    {
        if (tt_hit && ply > 0)
        {
            if (!is_pv)
            {
                return true;
            }
            else
            {

                if (flag == TT_EXACT ||
                    (flag == TT_BETA && tt_score >= beta))
                {
                    return true;
                }
            }
        }
        return false;
    }

    template <Color Us>
    inline void iterative_deepening(SearchWorker &worker, Move &tt_move, int depth, int ply, bool is_pv, int alpha, int beta)
    {
        if (tt_move != 0 || depth < 6 || !is_pv)
            return;

        int new_depth = depth - 4;
        worker.negamax<Us>(new_depth, alpha, beta, ply, true);
        tt_move = worker.get_tt().get_move(worker.get_board().get_hash());
    }

    template <Color Us>
    inline bool nmp(SearchWorker &worker, int depth, int ply, bool allow_null, bool in_check, bool is_mate_node, int alpha, int beta, int &return_score)
    {
        if (depth >= 3 && ply > 0 && allow_null && !in_check && !is_mate_node && beta < 9000 && alpha > -9000)
        {
            int stored_ep;
            worker.get_tt().prefetch(worker.get_board().get_hash());
            worker.get_board().play_null_move(stored_ep);
            int R = 2 + depth / 6;
            R = std::min(R, depth - 1);
            int score = -worker.negamax<!Us>(depth - 1 - R, -beta, -beta + 1, ply + 1, false);
            worker.get_board().unplay_null_move(stored_ep);

            if (score >= beta)
            {
                return_score = (score >= engine::config::eval::MateScore - engine::config::search::MaxDepth) ? beta : score;
                return true;
            }
        }
        return false;
    }

    template <Color Us>
    bool should_futility_pruning(const VBoard &board, int depth, int ply, bool in_check, bool is_mate_node, int alpha)
    {
        if (depth <= 3 && !in_check && ply > 0 && !is_mate_node)
        {
            int futil_margin = 175 + 110 * depth;
            int static_eval = Eval::lazy_eval_relative<Us>(board);

            if (static_eval + futil_margin <= alpha)
            {
                return true;
            }
        }
        return false;
    }

    inline void score_moves(SearchWorker &worker, MoveList &list, Move tt_move, int ply, int thread_id)
    {
        // Récupération du coup précédent pour les Counter-moves
        const Move prev_m = ply > 0 ? (worker.get_board().get_history()->back()).move : 0;

        uint64_t posKey = worker.get_board().get_hash() ^ (uint64_t(thread_id) << 32);

        if (thread_id != 0 && ply > 5)
            for (int i = 0; i < list.count; ++i)
            {
                int base = worker.score_move(list.moves[i], worker.get_board(), tt_move, ply, prev_m);

                uint64_t moveKey = (uint64_t)list.moves[i].get_value() * 0x9e3779b97f4a7c15ULL;
                uint64_t h = posKey ^ moveKey;

                uint64_t r = engine::random::splitmix64(h);

                int delta = std::max(1, std::abs(base) / 12);
                int noise = int(r & 1023) - 512; // [-512 .. 511]
                noise = noise * delta / 512;

                list.scores[i] = base + noise;
                list.is_tactical[i] = base >= engine::config::eval::TacticalScore;
            }
        else
            for (int i = 0; i < list.count; ++i)
            {
                list.scores[i] = worker.score_move(list.moves[i], worker.get_board(), tt_move, ply, prev_m);
                list.is_tactical[i] = list.scores[i] >= engine::config::eval::TacticalScore;
            }
    }
}

template <Color Us>
int SearchWorker::negamax(int depth, int alpha, int beta, int ply, bool allow_null)
{

    // =============================== Quick return cases ===============================
    if (check_stop())
        return alpha;

    if (search::is_null(board, ply))
        return (board.get_history_size() < 20) ? -25 : 0;

    const bool is_pv = (beta - alpha > 1);
    const bool in_check = board.is_king_attacked<Us>();
    const bool is_mate_node = (alpha < engine::config::eval::MateScore && beta > -engine::config::eval::MateScore && in_check);

    if (search::razoring<Us>(board, depth, alpha, is_pv, in_check))
        return qsearch<Us>(alpha, beta, ply);

    Move tt_move = 0;
    {
        TTFlag flag;
        int tt_score;
        bool tt_hit = shared_tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move, flag);
        if (search::should_use_tt(tt_hit, ply, is_pv, flag, tt_score, beta))
            return tt_score;
    }

    if (search::should_qsearch(depth, ply, in_check))
        return qsearch<Us>(alpha, beta, ply);

    if (search::reverse_futility_pruning<Us>(board, depth, ply, in_check, is_pv, beta))
        return beta;

    // =============================== Search ===============================

    search::iterative_deepening<Us>(*this, tt_move, depth, ply, is_pv, alpha, beta);

    {
        int return_score;
        if (search::nmp<Us>(*this, depth, ply, allow_null, in_check, is_mate_node, alpha, beta, return_score))
            return return_score;
    }

    const bool futil_pruning = search::should_futility_pruning<Us>(board, depth, ply, in_check, is_mate_node, alpha);

    // 6. Génération et tri des coups restants
    MoveList list;
    MoveGen::generate_pseudo_legal_moves<Us>(board, list);
    search::score_moves(*this, list, tt_move, ply, thread_id);

    // 7. PVS Loop (Principal Variation Search)
    int alpha_orig = alpha;

    int best_score = -engine::config::eval::Inf;
    Move best_move_this_node = 0;
    int moves_searched = 0;

    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);
        shared_tt.prefetch(board.get_hash_after(m));

        if (!board.is_move_legal(m))
            continue;

        int score;
        const bool is_tactical = list.is_tactical[i];

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
        if (!in_check && !is_pv &&
            depth <= 6 &&
            moves_searched > 1 &&
            m != tt_move &&
            m.is_capture())
        {
            if (Eval::get_piece_score(m.get_from_piece()) > Eval::get_piece_score(m.get_to_piece()))
            {
                int see_score = see(
                    m.get_to_sq(),
                    m.get_to_piece(),
                    m.get_from_piece(),
                    Us,
                    m.get_from_sq());

                int threshold = -15 * depth - Eval::get_piece_score(m.get_to_piece()) / 2;

                if (see_score < threshold)
                    continue;
            }
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
    if (ply == 0)
    {
        this->out_move = best_move_this_node;
    }

    // 8. Gestion des Mats et Pats
    if (moves_searched == 0)
    {
        int score = in_check ? -engine::config::eval::MateScore + ply : 0;
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