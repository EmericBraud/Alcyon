#include "gui.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    std::cout << b.load_fen("r3k2r/ppp2ppp/n1qp1n1b/3bp3/2B1P3/NPQ2N2/PBPP1PPP/R3K2R w KQkq - 0 1") << std::endl;
    std::cout << static_cast<int>(b.get_side_to_move()) << std::endl;
    GUI gui{b};
    gui.run();
    return 0;
}