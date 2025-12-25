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
    ASSERT_EQ(MoveGen::KingAttacks[Square::d4], 0b1110000010100000111000000000000000000ULL)
        << "Erreur : Les mouvements du roi sont invalides";

    const U64 bitboard_moves = MoveGen::KingAttacks[Square::b3];
    ASSERT_EQ(bitboard_moves, 0x7050700);
}

TEST_F(MoveGenTest, MaskGenRook)
{
    ASSERT_EQ(MoveGen::RookMasks[Square::d4], 0b1000000010000000100001110110000010000000100000000000ULL)
        << "Erreur : Le mask de lookup de la tour est invalide";
}

TEST_F(MoveGenTest, MaskGenBishop)
{
    ASSERT_EQ(MoveGen::BishopMasks[Square::d4], 0b1000000001000100001010000000000000101000010001000000000ULL)
        << "Erreur : Le mask de lookup du fou est invalide";
}

TEST_F(MoveGenTest, MoveGenPawn)
{
    ASSERT_EQ(MoveGen::PawnAttacksWhite[Square::d4], 0b1010000000000000000000000000000000000ULL)
        << "Erreur : Le mask d'attaque du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnAttacksBlack[Square::d4], 0b101000000000000000000ULL)
        << "Erreur : Le mask d'attaque du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPushWhite[Square::d4], 0b100000000000000000000000000000000000ULL)
        << "Erreur : Le mask de push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPushBlack[Square::d4], 0b10000000000000000000ULL)
        << "Erreur : Le mask de push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[Square::d4], 0b0ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[Square::d4], 0b0ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[Square::d2], 0b1000000000000000000000000000ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[Square::d2], 0b0ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";

    ASSERT_EQ(MoveGen::PawnPush2White[Square::d7], 0b0ULL)
        << "Erreur : Le mask de double push du pion blanc est invalide";
    ASSERT_EQ(MoveGen::PawnPush2Black[Square::d7], 0b100000000000000000000000000000000000ULL)
        << "Erreur : Le mask de double push du pion noir est invalide";
}

TEST_F(MoveGenTest, MoveGenRook)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R4/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_rook_moves(Square::d5, b);
    ASSERT_EQ(bitboard_moves, 0b1000000010001111011000001000000010000000100000000000);
}

TEST_F(MoveGenTest, MoveGenRookPieceOnBorder)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_rook_moves(Square::d5, b);
    ASSERT_EQ(bitboard_moves, 0b1000000010001111011000001000000010000000100000000000);
}

TEST_F(MoveGenTest, MoveGenBishop)
{
    Board b{};
    b.load_fen("8/1p1q4/5k2/1n1R4/8/1K2N1p1/3B4/8 w - - 0 1");

    // Half legal move (can capture his own pieces)
    const U64 bitboard_moves = MoveGen::generate_bishop_moves(Square::d2, b);
    ASSERT_EQ(bitboard_moves, 0b100000010000101000000000000010100);
}

TEST_F(MoveGenTest, MoveGenEnPassant)
{
    Board b;
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move move(Square::e2, Square::e4, PAWN);
    MoveGen::init_move_flags(b, move);
    b.play(move);
    U64 mask = MoveGen::get_legal_moves_mask(b, Square::f4);
    ASSERT_EQ(mask, 0x300000);
}

TEST_F(MoveGenTest, PawnCantPushThrought)
{
    Board b;
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/8/4p3/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    U64 mask = MoveGen::get_legal_moves_mask(b, Square::e2);
    ASSERT_EQ(mask, 0ULL);
}

TEST_F(MoveGenTest, PawnCanPush)
{
    Board b{};
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move move(Square::e2, Square::e4, PAWN);
    b.play(move);

    Move move2(Square::f4, Square::e3, PAWN);
    b.play(move2);
    U64 mask = MoveGen::get_legal_moves_mask(b, Square::d2);
    ASSERT_EQ(mask, 0x8180000);
}

TEST_F(MoveGenTest, EnPassantAfterCheckTest)
{
    Board b{};
    b.load_fen("rnbqkbnr/ppppp1pp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    U64 valid_mask = MoveGen::get_legal_moves_mask(b, Square::e2);
    ASSERT_EQ(valid_mask, 0x10100000);
    Move m{Square::e2, Square::e4, PAWN};
    MoveGen::init_move_flags(b, m);
    b.play(m);
    ASSERT_EQ(b.get_en_passant_sq(), Square::e3);
    valid_mask = MoveGen::get_legal_moves_mask(b, Square::f4);
    ASSERT_EQ(valid_mask, 0x300000);
    ASSERT_EQ(b.get_en_passant_sq(), Square::e3);
    /**
     * Here it needs a little explaination.
     * When we "get_legal_moves_mask",
     * under the hood we perform the move, get the opponent attack mask
     * then check if it collides with our king,
     * then undo the move. That is why we need to check if en_passant_sq
     * is still unchanged.
     */
    Move m2{Square::f4, Square::e3, PAWN};
    MoveGen::init_move_flags(b, m2);
    b.play(m2);
    ASSERT_EQ(b.get_piece_bitboard(WHITE, PAWN), 0xef00);
    ASSERT_EQ(b.get_piece_bitboard(BLACK, PAWN), 0xdf000000100000);
    ASSERT_EQ(b.get_en_passant_sq(), EN_PASSANT_SQ_NONE);
}

U64 Perft(Board &b, int depth)
{
    U64 nodes = 0;
    std::vector<Move> leg_moves = MoveGen::generate_legal_moves(b);
    if (depth == 1)
        return leg_moves.size();

    for (const Move &m : leg_moves)
    {
        b.play(m);
        nodes += Perft(b, depth - 1);
        b.unplay(m);
    }
    return nodes;
}

TEST_F(MoveGenTest, perft_test)
{
    Board b{};
    b.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 0");
    std::array<U64, 5> known_n_nodes = {48, 2039, 97862, 4085603, 193690690};

    for (int i{0}; i < 5; ++i)
    {
        const U64 n{Perft(b, i + 1)};
        ASSERT_EQ(n, known_n_nodes[i]);
    }
}