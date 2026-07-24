#include <bit>

#include "engine/eval/pos_eval.hpp"

#ifdef NNUE_EVAL

namespace Eval
{
    int eval(const VBoard &board, int alpha, int beta)
    {
        const int piece_count = std::popcount(board.get_occupancy<NO_COLOR>());
        const Color side_to_move = board.get_side_to_move();

        // Uses VBoard's incrementally-maintained accumulator (see
        // virtual_board.hpp's play/unplay hooks) -- no per-call recompute.
        // evaluate_abs() returns centipawns relative to the side to move (the
        // network is always evaluated as if "us" = whoever is on move). This
        // function's contract (matching the HCE eval and eval_relative<Us> in
        // pos_eval.hpp) is to return a WHITE-relative absolute score, so flip
        // the sign back for black to move.
        const int score = board.get_nnue_eval().evaluate_abs(side_to_move, piece_count);
        return side_to_move == WHITE ? score : -score; // @TODO create a templated version of it OR delete this statement (sign is flipped twice)
    }
}

#endif