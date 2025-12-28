#include "core/move/move_generator.hpp"
#include "gtest/gtest.h"

class CastlingTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        MoveGen::initialize_bitboard_tables();
    }
    void SetUp() override
    {
    }
};

TEST_F(CastlingTest, CanCastle)
{
    Board b;
    b.load_fen("r3k2r/pppqbppp/2npbn2/4p3/4P3/P1NPBN2/1PPQBPPP/R3K2R w KQkq - 1 9");
    MoveList list;
    MoveGen::generate_castle_moves(b, list);
    ASSERT_EQ(list.count, 2);
    Move m(Square::e1, Square::g1, Piece::KING, Move::Flags::KING_CASTLE);
    ASSERT_TRUE(b.get_castling_rights() & WHITE_KINGSIDE);
    b.play(m);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, KING), 0x40);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, ROOK), 0x21);
    ASSERT_TRUE(!(b.get_castling_rights() & WHITE_KINGSIDE));
}

TEST_F(CastlingTest, UndoCastle)
{
    Board b;
    b.load_fen("r3k2r/pppqbppp/2npbn2/4p3/4P3/P1NPBN2/1PPQBPPP/R3K2R w KQkq - 1 9");
    MoveList list;
    MoveGen::generate_castle_moves(b, list);
    Move m(Square::e1, Square::g1, Piece::KING, Move::Flags::KING_CASTLE, NO_PIECE, b.get_castling_rights());
    b.play(m);
    b.unplay(m);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, ROOK), 0x81);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, KING), 0x10);
    ASSERT_TRUE(b.get_castling_rights() & WHITE_KINGSIDE);
}