#include "board.hpp"
#include <iostream>

int main()
{
    Board b{};
    std::cout << b.load_fen("8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 b - - 99 50") << std::endl;
    b.show();
    return 0;
}