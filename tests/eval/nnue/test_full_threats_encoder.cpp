#include "gtest/gtest.h"
#include "engine/eval/nnue/full_threats_encoder.hpp"
#include "core/board/board.hpp"
#include "common/constants.hpp"
#include <vector>

// Verifies the runtime-built threat-offset table sums to exactly 60,720 (the
// NUM_REAL_FEATURES / static_assert(threatfeatures == 60720) from
// nnue-pytorch's training_data_loader.cpp), and that feature extraction on
// real positions only ever produces in-range, well-formed indices.

TEST(FullThreatsEncoderTest, OffsetTableTotalsExactly60720)
{
    const auto &table = nnue::threats::offset_table();
    EXPECT_EQ(table.total_features, 60720);
}

TEST(FullThreatsEncoderTest, StartingPositionProducesInRangeIndices)
{
    Board board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<int> white_features;
    std::vector<int> black_features;
    nnue::threats::fill_features<WHITE>(board, white_features);
    nnue::threats::fill_features<BLACK>(board, black_features);

    // The starting position has pawn-defends-pawn and knight/bishop attacks on
    // pawns (b1/g1 knights and c1/f1 bishops attack nothing beyond their own
    // pawns' squares... actually on the initial position, pieces mostly
    // attack their own side's pawns), so this should be non-empty.
    EXPECT_GT(white_features.size(), 0u);
    EXPECT_GT(black_features.size(), 0u);
    EXPECT_LE(white_features.size(), static_cast<std::size_t>(nnue::threats::NUM_INPUTS));

    for (int idx : white_features)
    {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, nnue::threats::NUM_INPUTS);
    }
    for (int idx : black_features)
    {
        EXPECT_GE(idx, 0);
        EXPECT_LT(idx, nnue::threats::NUM_INPUTS);
    }
}

TEST(FullThreatsEncoderTest, KingsNeverProduceFeaturesAsAttacker)
{
    // map[King][*] is all -1 in the source, so King is never a tracked
    // attacker. Verify via a position where kings are adjacent to other
    // pieces they could otherwise "attack".
    Board board;
    ASSERT_TRUE(board.load_fen("8/8/8/3k4/3K4/8/8/8 w - - 0 1"));

    std::vector<int> white_features;
    std::vector<int> black_features;
    nnue::threats::fill_features<WHITE>(board, white_features);
    nnue::threats::fill_features<BLACK>(board, black_features);

    EXPECT_TRUE(white_features.empty());
    EXPECT_TRUE(black_features.empty());
}

TEST(FullThreatsEncoderTest, KnightAttackingPawnProducesOneFeaturePerPerspective)
{
    // White knight on d5 attacks a black pawn on b6 (file d->b is -2, rank
    // 5->6 is +1: a valid knight move); simple sanity check that at least one
    // in-range feature is produced from both perspectives for a simple,
    // unambiguous threat.
    Board board;
    ASSERT_TRUE(board.load_fen("4k3/8/1p6/3N4/8/8/8/4K3 w - - 0 1"));

    std::vector<int> white_features;
    std::vector<int> black_features;
    nnue::threats::fill_features<WHITE>(board, white_features);
    nnue::threats::fill_features<BLACK>(board, black_features);

    EXPECT_GT(white_features.size(), 0u);
    EXPECT_GT(black_features.size(), 0u);
}
