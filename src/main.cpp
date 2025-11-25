#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    std::cout << b.load_fen(STARTING_POS_FEN) << std::endl;
    GUI gui{b};
    gui.run();
    return 0;
}