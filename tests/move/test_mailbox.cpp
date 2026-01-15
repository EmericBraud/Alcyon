#include "gtest/gtest.h"
#include "core/board/board.hpp"

class MailboxTest : public ::testing::Test
{
protected:
    Board b;
    const PieceInfo EMPTY = {Color::NO_COLOR, Piece::NO_PIECE};

    // Helper pour vérifier la cohérence Mailbox vs Paramètres
    void ExpectPiece(int sq, Color c, Piece p)
    {
        auto [res_c, res_p] = b.get_piece_on_square(sq);
        EXPECT_EQ(res_c, c) << "Mauvaise couleur sur la case " << sq;
        EXPECT_EQ(res_p, p) << "Mauvaise pièce sur la case " << sq;
    }
};

// 1. Test de l'initialisation FEN
TEST_F(MailboxTest, FEN_Initialisation)
{
    b.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Test coins et centre
    ExpectPiece(Square::a1, WHITE, ROOK);
    ExpectPiece(Square::e1, WHITE, KING);
    ExpectPiece(Square::d8, BLACK, QUEEN);
    ExpectPiece(Square::e4, NO_COLOR, NO_PIECE); // Case vide
}

// 2. Test du En Passant (Le point le plus critique)
TEST_F(MailboxTest, EnPassant_Mailbox_Sync)
{
    // Position où le pion blanc en d5 peut prendre en e6 après e7-e5
    b.load_fen("7k/8/8/3Pp3/8/8/8/7K w - e6 0 1");

    // Coup d5e6 (En Passant)
    Move ep_move(Square::d5, Square::e6, PAWN, Move::Flags::EN_PASSANT_CAP, PAWN);
    b.play(ep_move);

    // Vérifications après PLAY
    ExpectPiece(Square::d5, NO_COLOR, NO_PIECE); // Départ vide
    ExpectPiece(Square::e6, WHITE, PAWN);        // Arrivée occupée
    ExpectPiece(Square::e5, NO_COLOR, NO_PIECE); // Pion noir capturé DOIT être vide dans la mailbox

    b.unplay(ep_move);

    // Vérifications après UNPLAY
    ExpectPiece(Square::d5, WHITE, PAWN);        // Retour du blanc
    ExpectPiece(Square::e6, NO_COLOR, NO_PIECE); // e6 redevient vide
    ExpectPiece(Square::e5, BLACK, PAWN);        // Le pion noir capturé DOIT revenir
}

// 3. Test du Roque (Deux pièces bougent dans la mailbox)
TEST_F(MailboxTest, Castling_Mailbox_Sync)
{
    b.load_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");

    // Petit roque blanc (e1g1, h1f1)
    Move castle(Square::e1, Square::g1, KING, Move::Flags::KING_CASTLE);
    b.play(castle);

    ExpectPiece(Square::e1, NO_COLOR, NO_PIECE);
    ExpectPiece(Square::h1, NO_COLOR, NO_PIECE);
    ExpectPiece(Square::g1, WHITE, KING);
    ExpectPiece(Square::f1, WHITE, ROOK);

    b.unplay(castle);

    ExpectPiece(Square::e1, WHITE, KING);
    ExpectPiece(Square::h1, WHITE, ROOK);
    ExpectPiece(Square::g1, NO_COLOR, NO_PIECE);
    ExpectPiece(Square::f1, NO_COLOR, NO_PIECE);
}

// 4. Test de Promotion
TEST_F(MailboxTest, Promotion_Mailbox_Sync)
{
    b.load_fen("8/4P1k1/8/8/8/8/8/7K w - - 0 1");

    // e7e8=Queen
    Move promo(Square::e7, Square::e8, PAWN, Move::Flags::PROMOTION_MASK);
    b.play(promo);

    ExpectPiece(Square::e7, NO_COLOR, NO_PIECE);
    ExpectPiece(Square::e8, WHITE, QUEEN); // Doit être une Reine, pas un pion

    b.unplay(promo);

    ExpectPiece(Square::e7, WHITE, PAWN); // Doit redevenir un Pion
    ExpectPiece(Square::e8, NO_COLOR, NO_PIECE);
}