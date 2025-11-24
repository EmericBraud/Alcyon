#include "move_generator.hpp"
#include "gtest/gtest.h"

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

TEST_F(MoveGenTest, MoveGenRook)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R4/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_rook_moves(static_cast<int>(Square::d5), b);
    ASSERT_EQ(bitboard_moves, 0b1000000010001111011000001000000010000000100000000000);
}

TEST_F(MoveGenTest, MoveGenRookPieceOnBorder)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_rook_moves(static_cast<int>(Square::d5), b);
    ASSERT_EQ(bitboard_moves, 0b1000000010001111011000001000000010000000100000000000);
}

TEST_F(MoveGenTest, MoveGenBishop)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R4/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_bishop_moves(static_cast<int>(Square::d2), b);
    ASSERT_EQ(bitboard_moves, 0b100000010000101000000000000010100);
}