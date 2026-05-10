#include "gtest/gtest.h"

#include "engine/eval/virtual_board.hpp"

#ifdef NNUE_EVAL

TEST(NnueFeatureEncoderRegression, QueenPresenceChangesRawNnue)
{
    VBoard with_q;
    VBoard without_q;

    // Position A: black king e8, white king e1, black queen on a1
    ASSERT_TRUE(with_q.load_fen("4k3/8/8/8/8/8/8/q3K3 b - - 0 1"));

    // Position B: same but no black queen on a1
    ASSERT_TRUE(without_q.load_fen("4k3/8/8/8/8/8/8/4K3 b - - 0 1"));

    const int raw_a = with_q.get_nnue_eval().evaluate_abs();
    const int raw_b = without_q.get_nnue_eval().evaluate_abs();

    EXPECT_NE(raw_a, raw_b) << "NNUE raw output identical for positions that differ by a queen; feature encoder may collapse indices.";
}

#else

TEST(NnueFeatureEncoderRegression, RequiresNnueBuild)
{
    GTEST_SKIP() << "NNUE_EVAL not enabled in this build.";
}

#endif
