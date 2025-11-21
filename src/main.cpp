#include "move_generator.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    std::cout << b.load_fen(STARTING_POS_FEN) << std::endl;
    b.show();
    Move m(Square::a2, Square::c3, Piece::KNIGHT);
    b.play(m);
    b.show();

    return 0;
}