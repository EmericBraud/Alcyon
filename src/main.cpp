#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen("rnbqkbnr/pppp1ppp/8/8/5p2/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Move move(Square::e2, Square::e4, PAWN);
    b.play(move);

    Move move2(Square::f4, Square::e3, PAWN);
    b.play(move2);

    GUI gui{b};
    gui.run();
    return 0;
}