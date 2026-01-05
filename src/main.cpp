#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen("4r3/6k1/1p1p2p1/p1pPpr1p/PnP2P2/1Q1P3q/1P1B1PR1/R4K2 w - - 0 28");

    GUI gui{b, true, WHITE};
    gui.run();
    return 0;
}