#include "move_generator.hpp"
#include "gtest/gtest.h"

class MovePlayTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(MovePlayTest, Play)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1");
    Move move(Square::d5, Square::d6, ROOK);

    b.play(move);
    ASSERT_EQ(b.get_occupancy(NO_COLOR), 2859288583342080);
}

TEST_F(MovePlayTest, PlayUnplay)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1");
    Move move(Square::d5, Square::d6, ROOK);
    const bitboard occ_in = b.get_occupancy(NO_COLOR);
    const bitboard occ_white_in = b.get_occupancy(WHITE);
    const bitboard occ_white_rook_in = b.get_piece_bitboard(WHITE, ROOK);

    b.play(move);
    b.unplay(move);
    const bitboard occ_fi = b.get_occupancy(NO_COLOR);
    const bitboard occ_white_fi = b.get_occupancy(WHITE);
    const bitboard occ_white_rook_fi = b.get_piece_bitboard(WHITE, ROOK);

    ASSERT_EQ(occ_white_rook_in, occ_white_rook_fi);
    ASSERT_EQ(occ_white_in, occ_white_fi);
    ASSERT_EQ(occ_in, occ_fi);
}

TEST_F(MovePlayTest, PlayUnplayCapture)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1");
    Move move(Square::d5, Square::d7, ROOK);
    const bitboard occ_in = b.get_occupancy(NO_COLOR);
    const bitboard occ_white_in = b.get_occupancy(WHITE);
    const bitboard occ_white_rook_in = b.get_piece_bitboard(WHITE, ROOK);

    b.play(move);
    b.unplay(move);
    const bitboard occ_fi = b.get_occupancy(NO_COLOR);
    const bitboard occ_white_fi = b.get_occupancy(WHITE);
    const bitboard occ_white_rook_fi = b.get_piece_bitboard(WHITE, ROOK);

    ASSERT_EQ(occ_white_rook_in, occ_white_rook_fi);
    ASSERT_EQ(occ_white_in, occ_white_fi);
    ASSERT_EQ(occ_in, occ_fi);
}

TEST_F(MovePlayTest, EnPassant)
{
    Board b;
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move move(Square::e2, Square::e4, PAWN);
    b.play(move);
    ASSERT_EQ(move.get_flags(), Move::DOUBLE_PUSH);
    ASSERT_EQ(b.get_en_passant_sq(), Square::e3);
    Move move2(Square::f4, Square::e3, PAWN);
    b.play(move2);
    ASSERT_EQ(move2.get_flags(), Move::EN_PASSANT_CAP);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, PAWN), 0xef00);
    ASSERT_EQ(b.get_en_passant_sq(), EN_PASSANT_SQ_NONE);
}

TEST_F(MovePlayTest, UndoEnPassant)
{
    Board b;
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move move(Square::e2, Square::e4, PAWN);
    b.play(move);
    Move move2(Square::f4, Square::e3, PAWN);
    b.play(move2);
    b.unplay(move2);
    ASSERT_EQ(b.get_piece_bitboard(BLACK, PAWN), 0xef000020000000);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, PAWN), 0x1000ef00);
    ASSERT_EQ(b.get_en_passant_sq(), Square::e3);
}

TEST_F(MovePlayTest, QueenPromotionDoAndUndo)
{
    Board b;
    b.load_fen("4k3/2P5/8/8/8/8/8/4K3 w - - 0 1");
    Move move(Square::c7, Square::c8, PAWN);
    b.play(move);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, PAWN), 0ULL);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, QUEEN), sq_mask(Square::c8));
    b.unplay(move);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, PAWN), sq_mask(Square::c7));
    ASSERT_EQ(b.get_piece_bitboard(WHITE, QUEEN), 0ULL);
}