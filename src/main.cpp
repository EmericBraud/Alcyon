#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen("5kr1/1p3p2/5p2/1P2qP1K/P1Bp3P/4bQ2/2P1R3/8 w - - 1 44");

    GUI gui{b, true, WHITE};
    gui.run();
    return 0;
}