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

static int expected_feature_index(Color perspective, int king_sq, Color piece_color, Piece piece_type, int piece_square)
{
    const int flip = (perspective == WHITE) ? 0 : 56;
    const int oriented_king_sq = king_sq ^ flip;
    const int oriented_piece_sq = piece_square ^ flip;

    const int king_file = oriented_king_sq % 8;
    const int mirrored_file = (king_file < 4) ? king_file : (7 - king_file);
    const int king_rank = oriented_king_sq / 8;
    const int king_bucket = (7 - king_rank) * 4 + mirrored_file;

    const int piece_idx = (piece_type == KING)
                              ? 10
                              : static_cast<int>(piece_type) * 2 + (piece_color != perspective ? 1 : 0);

    int final_piece_sq = oriented_piece_sq;
    if (king_file < 4)
        final_piece_sq ^= 7;

    return (king_bucket * 11 * 64) + (piece_idx * 64) + final_piece_sq;
}

TEST_F(FeatureEncoderTest, KingShouldReturnCorrectValue)
{
    const int king_sq = Square::a3;
    const int piece_square = Square::e4;
    const int expected_idx = expected_feature_index(WHITE, king_sq, WHITE, KING, piece_square);

    const int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, KING, piece_square);
    ASSERT_EQ(expected_idx, idx);
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

    int expected_idx = expected_feature_index(WHITE, king_sq, WHITE, piece_type, piece_square);

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenEnemyPiece)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;

    int expected_idx = expected_feature_index(WHITE, king_sq, BLACK, piece_type, piece_square);

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldReturnCorrectValueWhenBlackSide)
{
    int king_sq = Square::a3;
    int piece_square = Square::e4;
    Piece piece_type = QUEEN;

    int expected_idx = expected_feature_index(BLACK, king_sq, BLACK, piece_type, piece_square);

    int idx = feature_encoder::get_feature_index<BLACK>(king_sq, BLACK, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}

TEST_F(FeatureEncoderTest, ShouldMirrorCorrectly)
{
    int king_sq = Square::g1;
    int piece_square = Square::h3;
    Piece piece_type = PAWN;

    int expected_idx = expected_feature_index(WHITE, king_sq, WHITE, piece_type, piece_square);

    int idx = feature_encoder::get_feature_index<WHITE>(king_sq, WHITE, piece_type, piece_square);
    ASSERT_EQ(expected_idx, idx);
}