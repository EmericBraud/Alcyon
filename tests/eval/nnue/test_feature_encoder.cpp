#include "gtest/gtest.h"

#include <type_traits>

#include "common/constants.hpp"
#include "core/piece/piece.hpp"
#include "core/piece/color.hpp"
#include "engine/eval/nnue/features_encoder.hpp"

class FeatureEncoderTest : public ::testing::Test
{
};

template <Color C>
concept HasFeatureIndex = requires {
    feature_encoder::get_feature_index<C>(
        static_cast<int>(Square::a3),
        WHITE,
        PAWN,
        static_cast<int>(Square::a3));
};

TEST_F(FeatureEncoderTest, KingShouldThrow)
{
    ASSERT_DEATH(feature_encoder::get_feature_index<WHITE>(Square::a3, WHITE, KING, Square::a3), ".*");
}

TEST_F(FeatureEncoderTest, NoColorShouldNotBeCallable)
{
    static_assert(!HasFeatureIndex<NO_COLOR>);
    static_assert(HasFeatureIndex<WHITE>);
    static_assert(HasFeatureIndex<BLACK>);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenAllyPiece)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;
    int expected_idx = (king_sq * 10 + piece_type) * 64 + piece_square;
    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenEnemyPiece)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;
    int expected_idx = (king_sq * 10 + piece_type + 5) * 64 + piece_square;
    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenBlackSide)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;
    int expected_idx = ((king_sq ^ 56) * 10 + piece_type) * 64 + (piece_square ^ 56);
    int idx = feature_encoder::get_feature_index<BLACK>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}