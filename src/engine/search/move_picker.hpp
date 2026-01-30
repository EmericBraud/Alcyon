#pragma once

#include <cstdint>

#include "common/constants.hpp"
#include "core/move/move.hpp"
#include "core/move/move_list.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/eval/virtual_board.hpp"
#include "worker.hpp"

enum PickerStages : uint8_t
{
    TT,
    PROMOTIONS,
    GOOD_CAPTURES,
    KILLERS,
    COUNTERS,
    QUIETS,
    BAD_CAPTURES,
    END
};
struct MovePicker
{
    alignas(64) MoveList list;
    alignas(64) uint8_t stage;
    alignas(64) int index;
    alignas(64) int bad_captures_index;
    int ply;
    Move prev_move;
    Move tt_move;
    int thread_id;

    // NOUVEAU : On stocke l'info ici pour que negamax la lise en toute sécurité
    bool current_is_tactical;

    MovePicker(VBoard &board, Move _tt_move, int _ply, Move _prev_move, int _thread_id)
    {
        list.clear();
        stage = TT;
        index = 0;
        ply = _ply;
        prev_move = _prev_move;
        tt_move = _tt_move;
        thread_id = _thread_id;
        current_is_tactical = true;
        bad_captures_index = constants::MaxMoves - 1;
        if (tt_move != 0 && board.is_move_pseudo_legal(tt_move))
        {
            MoveGen::init_move_flags(board, tt_move);
            list.push(tt_move, tt_move.is_capture() || tt_move.is_promotion());
            current_is_tactical = tt_move.is_capture() || tt_move.is_promotion();
        }
    }

    template <Color Us>
    Move pick_next(SearchWorker &worker)
    {
        VBoard &board = worker.get_board();

        // 1. Si le stage est fini, on passe au suivant
        if (stage != END && index >= list.count)
        {
            ++stage;
            index = 0;
            list.count = 0;
            int i = 0;
            Move m1, m2, counter_move;

            switch (stage)
            {
            case PROMOTIONS:
                MoveGen::generate_pseudo_legal_promotions<Us>(board, list);

                for (int j = 0; j < list.size(); ++j)
                {
                    const Move m = list[j];
                    if (m == tt_move)
                    {
                        continue;
                    }

                    int score = 20000 + Eval::get_piece_score(m.get_promo_piece());

                    list.scores[i] = score;
                    list.is_tactical[i] = true;
                    list[i++] = list[j];
                }
                list.count = i;
                break;

            case GOOD_CAPTURES:
                MoveGen::generate_pseudo_legal_captures<Us>(board, list);
                for (int j = 0; j < list.size(); ++j)
                {
                    const Move m = list[j];
                    if (m == tt_move)
                        continue;
                    if (m.is_promotion())
                        continue;

                    const int victim = m.get_to_piece();
                    const int attacker = m.get_from_piece();

                    const int mvv_score = engine_constants::eval::MvvLvaTable[victim][attacker];

                    int final_score;

                    if (victim > attacker)
                    {
                        final_score = 10000 + mvv_score;
                    }
                    else
                    {
                        int see_val = worker.see<Us>(m.get_to_sq(), static_cast<Piece>(victim), static_cast<Piece>(attacker), m.get_from_sq());

                        if (see_val >= 0)
                        {
                            final_score = 8000 + mvv_score;
                        }
                        else
                        {
                            list[bad_captures_index] = m;
                            list.is_tactical[bad_captures_index] = true;
                            list.scores[bad_captures_index--] = see_val;
                            continue;
                        }
                    }

                    list.scores[i] = final_score;
                    list.is_tactical[i] = true;
                    list[i++] = m;
                }
                list.count = i;
                break;

            case KILLERS:
                m1 = worker.killer_moves[ply][0];
                m2 = worker.killer_moves[ply][1];
                MoveGen::init_move_flags(board, m1);
                MoveGen::init_move_flags(board, m2);

                if (m1 != 0 && m1 != tt_move && board.is_move_pseudo_legal(m1) && !m1.is_capture() && !m1.is_promotion())
                {
                    list.scores[i] = 8000;
                    list.is_tactical[i] = false;
                    list[i++] = m1;
                }
                if (m2 != 0 && m2 != tt_move && m2 != m1 && board.is_move_pseudo_legal(m2) && !m2.is_capture() && !m2.is_promotion())
                {
                    list.scores[i] = 7900;
                    list.is_tactical[i] = false;
                    list[i++] = m2;
                }
                list.count = i;
                break;

            case COUNTERS:
                if (ply > 0 && prev_move != 0)
                {
                    counter_move = worker.counter_moves[Us][prev_move.get_from_piece()][prev_move.get_to_sq()];
                    MoveGen::init_move_flags(board, counter_move);
                    if (counter_move != 0 && counter_move != tt_move &&
                        counter_move != worker.killer_moves[ply][0] &&
                        counter_move != worker.killer_moves[ply][1] &&
                        !counter_move.is_capture() &&
                        board.is_move_pseudo_legal(counter_move) &&
                        !counter_move.is_promotion())
                    {
                        list.scores[i] = 7500;
                        list.is_tactical[i] = false;
                        list[i++] = counter_move;
                    }
                }
                list.count = i;
                break;

            case QUIETS:
                MoveGen::generate_pseudo_legal_moves<Us>(board, list);
                for (int j = 0; j < list.size(); ++j)
                {
                    const Move m = list[j];
                    if (m.is_capture() || m == tt_move ||
                        m.is_promotion() ||
                        m == worker.killer_moves[ply][0] || m == worker.killer_moves[ply][1] ||
                        (ply > 0 && prev_move != 0 && m == worker.counter_moves[Us][prev_move.get_from_piece()][prev_move.get_to_sq()]))
                        continue;
                    int noise = 0;
                    if (thread_id != 0)
                    {
                        uint64_t hash = (uint64_t(m.get_value()) + ply) ^ (uint64_t(thread_id) << 32);
                        noise = (hash & 0x7FF) - 1024;
                    }
                    list.scores[i] = worker.history_moves[Us][m.get_from_sq()][m.get_to_sq()] + noise;
                    list.is_tactical[i] = false;
                    list[i++] = m;
                }
                list.count = i;
                break;

            case BAD_CAPTURES:
                // A little trick I found to avoid processing the SEE twice while keeping a single list ;)
                index = bad_captures_index + 1;
                list.count = constants::MaxMoves;
                break;
            case END:
                return 0;
            }
            return pick_next<Us>(worker);
        }

        if (stage == END)
            return 0;

        Move best = list.pick_best_move(index);

        current_is_tactical = list.is_tactical[index];

        index++;
        return best;
    }
};