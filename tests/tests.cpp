#include "board.hpp"
#include "gtest/gtest.h"

class BoardTest : public ::testing::Test
{
protected:
    Board b;

    void SetUp() override
    {
    }
};

// 2. Utiliser la macro TEST_F pour définir le test unitaire
// TestSuiteName: BoardTest (le nom de la Fixture)
// TestName: KnightMove_Initial
TEST_F(BoardTest, KnightMove_Initial)
{

    b.load_fen(STARTING_POS_FEN);

    Move m(Square::a2, Square::c3, Piece::KNIGHT);
    b.play(m);

    PieceInfo empty_square = std::make_pair(Color::NO_COLOR, Piece::NO_PIECE);
    PieceInfo knight_on_c3 = std::make_pair(Color::WHITE, Piece::KNIGHT);

    // B. Act (Action)
    // Ici, nous supposons que la fonction apply_move() ou play() est appelée.
    // Votre code n'appelle pas de méthode de déplacement, mais nous allons simuler
    // l'état post-déplacement pour valider les assertions.

    // NOTE: Si le test est censé valider l'objet Move sans l'appliquer,
    // l'objet Board n'est pas modifié et les assertions échoueront.
    // Supposons que vous ayez une méthode pour appliquer le mouvement.
    // b.apply_move(m);

    // Pour l'exercice, si on simule l'état (le test doit être revu si l'état n'est pas modifié) :
    // b.set_piece_on_square(Square::c3, Color::WHITE, Piece::KNIGHT);
    // b.set_piece_on_square(Square::a2, Color::NO_COLOR, Piece::NO_PIECE);

    // C. Assert (Vérification)
    // Vérifie la pièce déplacée (supposée être appliquée à la ligne précédente)
    // Note: Le cast statique vers int n'est pas nécessaire si vous utilisez Square directement
    // et que get_piece_on_square est surchargée pour accepter Square.
    ASSERT_EQ(b.get_piece_on_square(Square::c3), knight_on_c3)
        << "Erreur : Le Cavalier Blanc n'est pas sur c3 après le mouvement.";

    // Vérifie l'ancienne case (doit être vide)
    ASSERT_EQ(b.get_piece_on_square(Square::a2), empty_square)
        << "Erreur : La case de départ a2 n'est pas vide.";
}