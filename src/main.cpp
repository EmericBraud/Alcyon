#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    std::cout << b.load_fen("8/1p1q4/5k2/1n1R3r/8/1K2N1p1/3B4/8 w - - 0 1") << std::endl;
    std::cout << static_cast<int>(b.get_side_to_move()) << std::endl;
    GUI gui{b};
    gui.run();
    return 0;
}