#include "core/board.hpp"
#include "gtest/gtest.h"

class BoardTest : public ::testing::Test
{
protected:
    Board b;

    void SetUp() override
    {
    }
};

TEST_F(BoardTest, KnightMove_Initial)
{

    b.load_fen(core::constants::FenInitPos);

    Move m(Square::b1, Square::c3, Piece::KNIGHT);
    b.play(m);

    PieceInfo empty_square = std::make_pair(Color::NO_COLOR, Piece::NO_PIECE);
    PieceInfo knight_on_c3 = std::make_pair(Color::WHITE, Piece::KNIGHT);

    ASSERT_EQ(b.get_piece_on_square(Square::c3), knight_on_c3)
        << "Erreur : Le Cavalier Blanc n'est pas sur c3 après le mouvement.";

    // Vérifie l'ancienne case (doit être vide)
    ASSERT_EQ(b.get_piece_on_square(Square::b1), empty_square)
        << "Erreur : La case de départ a2 n'est pas vide.";
}