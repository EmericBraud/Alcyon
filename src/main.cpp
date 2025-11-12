#include "board.hpp"
#include <iostream>
#include <string>

int main()
{
    Board b{};
    const std::string starting_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    std::cout << b.load_fen(starting_fen) << std::endl;
    b.show();
    return 0;
}