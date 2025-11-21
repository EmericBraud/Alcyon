#include "board.hpp"

int main()
{
    Board b{};
    std::cout << b.load_fen(STARTING_POS_FEN) << std::endl;
    b.show();
    Move m(Square::a2, Square::c3, Piece::KNIGHT);
    b.play(m);
    b.show();

    PieceInfo empty_square = std::make_pair(Color::NO_COLOR, Piece::NO_PIECE);
    PieceInfo knight_on_c3 = std::make_pair(Color::WHITE, Piece::KNIGHT);

    return 0;
}