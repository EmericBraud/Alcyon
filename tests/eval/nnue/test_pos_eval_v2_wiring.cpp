#include <cstdlib>

#include "gtest/gtest.h"

#include "common/constants.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/eval/virtual_board.hpp"

// Exercises Eval::eval()'s eval-command call site (mirrors what
// UCI::run_eval() does via Eval::eval_relative<Us>) with the NNUE_EVAL_V2
// compile-time feature flag. When the flag is off (default), this simply
// confirms the existing v1 path keeps working unchanged. When the flag is on
// (ENABLE_NNUE_EVAL_V2=ON at configure time), it confirms the from-scratch
// NnueEvalV2 path (full_threats_encoder + halfka_v2_hm_encoder feature
// recompute on every call, no incremental reuse) produces sane, deterministic,
// WHITE-relative scores on a few standard positions without crashing.
namespace
{
    struct EvalCase
    {
        const char *name;
        const char *fen;
    };

    const EvalCase kCases[] = {
        {"StartPos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        // Tactical middlegame position (pinned knight, open lines).
        {"Tactical", "r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1"},
        // Simple king-and-pawn endgame.
        {"Endgame", "8/5k2/8/8/8/3K4/4P3/8 w - - 0 1"},
    };
}

class PosEvalV2WiringTest : public ::testing::TestWithParam<EvalCase>
{
};

TEST_P(PosEvalV2WiringTest, EvalDoesNotCrashAndIsDeterministic)
{
    const EvalCase &c = GetParam();

    VBoard board;
    ASSERT_TRUE(board.load_fen(c.fen));

    const int score_a = Eval::eval(board, -engine_constants::eval::Inf, engine_constants::eval::Inf);
    const int score_b = Eval::eval(board, -engine_constants::eval::Inf, engine_constants::eval::Inf);

    // Deterministic: re-evaluating the same (unchanged) position must give
    // the exact same score, whether that's the v1 incremental accumulator or
    // v2's from-scratch recompute.
    EXPECT_EQ(score_a, score_b);

    // Sane bound: none of these positions should produce a wild/out-of-range
    // score (which would indicate a scale/sign/wiring bug), NNUE mate scores
    // aside.
    EXPECT_LT(std::abs(score_a), engine_constants::eval::MateScore);
}

INSTANTIATE_TEST_SUITE_P(
    StandardPositions,
    PosEvalV2WiringTest,
    ::testing::ValuesIn(kCases),
    [](const ::testing::TestParamInfo<EvalCase> &info)
    { return info.param.name; });

TEST(PosEvalV2WiringTest, WhiteBlackSignConventionMatches)
{
    // Same position, side to move flipped by a null-ish setup: use two FENs
    // that are mirror images of each other (colors swapped, side to move
    // swapped) to confirm the WHITE-relative absolute score contract holds
    // regardless of which NNUE version (v1 or v2) is wired in.
    VBoard white_to_move;
    ASSERT_TRUE(white_to_move.load_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"));
    VBoard black_to_move;
    ASSERT_TRUE(black_to_move.load_fen("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1"));

    const int score_white_pov = Eval::eval(white_to_move, -engine_constants::eval::Inf, engine_constants::eval::Inf);
    const int score_black_pov = Eval::eval(black_to_move, -engine_constants::eval::Inf, engine_constants::eval::Inf);

    // White has an extra pawn on e2 in the first position: WHITE-relative
    // score should be positive. In the mirrored position black has the
    // extra pawn: WHITE-relative score should be negative.
    EXPECT_GT(score_white_pov, 0);
    EXPECT_LT(score_black_pov, 0);
}
