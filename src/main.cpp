#include "interface/uci.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    UCI u;
    u.loop();
    return 0;
}