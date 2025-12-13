#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen("3pkp2/3ppp2/1Rq5/8/8/8/3PPP2/3PKP2 b - - 0 1");

    GUI gui{b};
    gui.run();
    return 0;
}