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

    int expected_ksq = king_sq;
    int expected_psq = piece_square;
    if ((expected_ksq % 8) > 3)
    {
        expected_ksq ^= 7;
        expected_psq ^= 7;
    }
    int king_mapped_idx = (expected_ksq / 8) * 4 + (expected_ksq % 8);
    int expected_idx = (king_mapped_idx * 12 * 64) + (static_cast<int>(piece_type) * 64) + expected_psq;

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenEnemyPiece)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;

    int expected_ksq = king_sq;
    int expected_psq = piece_square;
    if ((expected_ksq % 8) > 3)
    {
        expected_ksq ^= 7;
        expected_psq ^= 7;
    }
    int king_mapped_idx = (expected_ksq / 8) * 4 + (expected_ksq % 8);
    int expected_idx = (king_mapped_idx * 12 * 64) + ((static_cast<int>(piece_type) + 6) * 64) + expected_psq;

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenBlackSide)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;

    int expected_ksq = king_sq ^ 56;
    int expected_psq = piece_square ^ 56;
    if ((expected_ksq % 8) > 3)
    {
        expected_ksq ^= 7;
        expected_psq ^= 7;
    }
    int king_mapped_idx = (expected_ksq / 8) * 4 + (expected_ksq % 8);
    int expected_idx = (king_mapped_idx * 12 * 64) + (static_cast<int>(piece_type) * 64) + expected_psq;

    int idx = feature_encoder::get_feature_index<BLACK>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldMirrorCorrectly)
{
    int king_sq = Square::g1;
    int piece_square = Square::h3;
    Piece piece_type = PAWN;

    int mirrored_ksq = Square::b1;
    int mirrored_psq = Square::a3;
    int king_mapped_idx = (mirrored_ksq / 8) * 4 + (mirrored_ksq % 8);
    int expected_idx = (king_mapped_idx * 12 * 64) + (static_cast<int>(piece_type) * 64) + mirrored_psq;

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}