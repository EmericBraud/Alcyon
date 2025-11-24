#include "move_generator.hpp"

int main()
{
    MoveGen::initialize_bitboard_tables();
    Board b{};
    std::cout << b.load_fen(STARTING_POS_FEN) << std::endl;
    b.show();
    std::vector<Move> v{};
    v = MoveGen::generate_pseudo_legal_moves(b, b.get_side_to_move());
    std::cout << v.size() << std::endl;
    return 0;
}