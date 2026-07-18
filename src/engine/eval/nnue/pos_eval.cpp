#include <bit>

#include "engine/eval/pos_eval.hpp"

#ifdef NNUE_EVAL_V2
#include "common/file.hpp"
#include "engine/eval/nnue/v2/nnue_eval_v2.hpp"
#endif

namespace Eval
{
#ifdef NNUE_EVAL_V2
    namespace
    {
        // Lazily-loaded, process-wide NNUE v2 model. Mirrors the lazy/cached
        // loading spirit of VBoard's v1 NnueEval member, but v2 doesn't yet
        // support incremental updates (see nnue_eval_v2.hpp scope note), so
        // instead of living inside VBoard it is a standalone instance that is
        // fully re-initialized (full feature recompute) from the board on
        // every call to eval().
        nnue_v2::NnueEvalV2 &get_nnue_eval_v2()
        {
            static nnue_v2::NnueEvalV2 instance(file::get_data_path("nnue/v2.nnue"));
            return instance;
        }
    }
#endif

    int eval(const VBoard &board, int alpha, int beta)
    {
        const int piece_count = std::popcount(board.get_occupancy<NO_COLOR>());
        const Color side_to_move = board.get_side_to_move();

#ifdef NNUE_EVAL_V2
        // Full recompute each call (no incremental accumulator support yet
        // for v2 -- a known follow-up). Same WHITE-relative absolute score
        // contract as the v1 path below.
        nnue_v2::NnueEvalV2 &nnue_v2_eval = get_nnue_eval_v2();
        nnue_v2_eval.initialize(board);
        const int score_v2 = nnue_v2_eval.evaluate_abs(side_to_move, piece_count);
        return side_to_move == WHITE ? score_v2 : -score_v2;
#else
        // evaluate_abs() returns centipawns relative to the side to move (the
        // network is always evaluated as if "us" = whoever is on move). This
        // function's contract (matching the HCE eval and eval_relative<Us> in
        // pos_eval.hpp) is to return a WHITE-relative absolute score, so flip
        // the sign back for black to move.
        const int score = board.get_nnue_eval().evaluate_abs(side_to_move, piece_count);
        return side_to_move == WHITE ? score : -score;
#endif
    }
}