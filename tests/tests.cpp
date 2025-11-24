#include "move_generator.hpp"
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

    b.load_fen(STARTING_POS_FEN);

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

class MoveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(MoveTest, MoveDescriptor)
{
    Move m{Square::a2, Square::c3, KNIGHT};

    ASSERT_EQ(m.get_from_piece(), KNIGHT);
    ASSERT_EQ(m.get_from_sq(), static_cast<uint32_t>(Square::a2));
    ASSERT_EQ(m.get_to_sq(), static_cast<uint32_t>(Square::c3));
    ASSERT_EQ(m.is_castling(), false);
    ASSERT_EQ(m.is_promotion(), false);
}

class MoveGenTest : public ::testing::Test
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

TEST_F(MoveGenTest, MoveGenKing)
{
    ASSERT_EQ(MoveGen::KingAttacks[static_cast<int>(Square::d4)], 0b1110000010100000111000000000000000000ULL)
        << "Erreur : Les mouvements du roi sont invalides";
}

TEST_F(MoveGenTest, MaskGenRook)
{
    ASSERT_EQ(MoveGen::RookMasks[static_cast<int>(Square::d4)], 0b1000000010000000100001110110000010000000100000000000ULL)
        << "Erreur : Le mask de lookup de la tour est invalide";
}

TEST_F(MoveGenTest, MaskGenBishop)
{
    ASSERT_EQ(MoveGen::BishopMasks[static_cast<int>(Square::d4)], 0b1000000001000100001010000000000000101000010001000000000ULL)
        << "Erreur : Le mask de lookup du fou est invalide";
}

TEST_F(MoveGenTest, MoveGenPawn)
{
    ASSERT_EQ(MoveGen::PawnAttacksWhite[static_cast<int>(Square::d4)], 0b1010000000000000000000000000000000000ULL)
        << "Erreur : Le mask d'attaque du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnAttacksBlack[static_cast<int>(Square::d4)], 0b101000000000000000000ULL)
        << "Erreur : Le mask d'attaque du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPushWhite[static_cast<int>(Square::d4)], 0b100000000000000000000000000000000000ULL)
        << "Erreur : Le mask de push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPushBlack[static_cast<int>(Square::d4)], 0b10000000000000000000ULL)
        << "Erreur : Le mask de push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[static_cast<int>(Square::d4)], 0b0ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[static_cast<int>(Square::d4)], 0b0ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[static_cast<int>(Square::d2)], 0b1000000000000000000000000000ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[static_cast<int>(Square::d2)], 0b0ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[static_cast<int>(Square::d7)], 0b0ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[static_cast<int>(Square::d7)], 0b100000000000000000000000000000000000ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";
}
