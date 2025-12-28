#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    b.load_fen(FEN_INIT_POS);

    GUI gui{b, true, WHITE};
    gui.run();
    return 0;
}