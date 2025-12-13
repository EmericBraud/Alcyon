#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    GUI gui{b};
    gui.run();
    return 0;
}