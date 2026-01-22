#include "worker.hpp"

#include "engine/utils/random.hpp"
#include "engine/engine_manager.hpp"
#include "engine/search/move_picker.hpp"

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
}

template <Color Us>
int SearchWorker::negamax(int depth, int alpha, int beta, int ply, bool allow_null, Move excluded_move)
{

    // =============================== Quick return cases ===============================
    if (check_stop())
        return alpha;

    if (max_extended_depth < ply)
        max_extended_depth = ply;

    if (search::is_null(board, ply))
        return (board.get_history_size() < 20) ? -25 : 0;

    if (ply >= engine::config::search::MaxDepth)
        return Eval::lazy_eval_relative<Us>(board);

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
        if (tt_move != excluded_move && search::should_use_tt(tt_hit, ply, is_pv, flag, tt_score, beta))
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

    const Move prev_m = ply > 0 ? (board.get_history()->back()).move : 0;
    MovePicker list(board, tt_move, ply, prev_m, thread_id);

    // 7. PVS Loop (Principal Variation Search)
    int alpha_orig = alpha;

    int best_score = -engine::config::eval::Inf;
    Move best_move_this_node = 0;
    int moves_searched = 0;

    while (true)
    {
        Move m = list.pick_next<Us>(*this);

        if (m == 0) // No moves left
            break;

        if (!board.is_move_legal<Us>(m))
            continue;

        if (m == excluded_move)
            continue;

        shared_tt.prefetch(board.get_hash_after(m));
        bool is_singular = false;

        if (m == tt_move && depth >= 8 && ply > 0 && excluded_move == 0 && !in_check)
        {
            TTFlag ttf;
            int tts;
            Move ttm;
            if (shared_tt.probe(board.get_hash(), depth, ply, -engine::config::eval::Inf, engine::config::eval::Inf, tts, ttm, ttf))
            {
                if (ttf == TT_EXACT || ttf == TT_ALPHA)
                {
                    int singular_beta = tts - (depth * 2);
                    int singular_depth = (depth - 1) / 2;
                    int score = negamax<Us>(singular_depth, singular_beta - 1, singular_beta, ply, false, m);

                    if (score < singular_beta)
                    {
                        is_singular = true;
                    }
                }
            }
        }

        int score;
        const bool is_tactical = list.current_is_tactical;
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
                int see_score = see<Us>(
                    m.get_to_sq(),
                    m.get_to_piece(),
                    m.get_from_piece(),
                    m.get_from_sq());

                int threshold = -15 * depth - Eval::get_piece_score(m.get_to_piece()) / 2;

                if (see_score < threshold)
                    continue;
            }
        }

        ++moves_searched;

        board.play<Us>(m);

        bool gives_check = board.is_king_attacked<!Us>();

        int extension = 0;
        if (gives_check && depth >= 2)
            extension = 1;
        if (is_singular)
            extension = 1;
        int new_depth = depth - 1 + extension;

        if (ply + new_depth >= engine::config::search::MaxDepth)
            new_depth = engine::config::search::MaxDepth - ply;

        // --- LATE MOVE REDUCTION (LMR) ---
        if (depth >= 3 && moves_searched > 4 && !is_tactical && !in_check && extension == 0)
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
            score = -negamax<!Us>(new_depth, -alpha - 1, -alpha, ply + 1, true);
        }
        else // Full Window Search (seulement si moves_searched == 1 et pas de TT move)
        {
            score = -negamax<!Us>(new_depth, -beta, -alpha, ply + 1, true);
        }

        // Si le score est dans la fenêtre mais pas une coupure, on re-cherche normalement
        if (score > alpha && score < beta && moves_searched > 1)
        {
            score = -negamax<!Us>(new_depth, -beta, -alpha, ply + 1, true);
        }

        board.unplay<Us>(m);

        // --- MISE À JOUR DES SCORES ET DES TABLES ---
        if (score >= beta)
        {
            shared_tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {

                // MALUS : On punit tous les coups calmes testés AVANT et qui ont échoué
                if (list.stage == PickerStages::QUIETS)
                {
                    int bonus = depth * depth;

                    // On récompense le coup gagnant
                    history_moves[Us][m.get_from_sq()][m.get_to_sq()] += bonus;
                    for (int j = 0; j < list.index - 1; ++j)
                    {
                        Move failed_move = list.list.moves[j];
                        // On ne punit que les coups calmes (pas les captures/promotions)

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

template int SearchWorker::negamax<WHITE>(int depth, int alpha, int beta, int ply, bool allow_null, Move excluded_move);
template int SearchWorker::negamax<BLACK>(int depth, int alpha, int beta, int ply, bool allow_null, Move excluded_move);