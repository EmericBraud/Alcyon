#include "engine/search/worker.hpp"

#include "core/move/generator/move_generator.hpp"

template <Color Side>
int SearchWorker::see(int sq, Piece target, Piece attacker, int from_sq) const
{
    int value[32];
    int d = 0;
    U64 occupied = board.get_occupancy(NO_COLOR);

    value[0] = Eval::get_piece_score(target);
    Piece victim = attacker;

    U64 all_attackers = MoveGen::attackers_to(sq, occupied, board);

    occupied ^= (1ULL << from_sq);
    all_attackers |= MoveGen::update_xrays(sq, occupied, board);

    Color current_side = !Side;
    Piece next_attacker = NO_PIECE;

    while (true)
    {
        d++;
        U64 potential_attackers = all_attackers & occupied;
        int from = current_side == WHITE ? board.get_smallest_attacker<WHITE>(potential_attackers, next_attacker) : board.get_smallest_attacker<BLACK>(potential_attackers, next_attacker);

        if (from == -1)
            break;

        value[d] = Eval::get_piece_score(victim) - value[d - 1];

        if (std::max(-value[d - 1], value[d]) < 0)
            break;

        occupied ^= (1ULL << from);
        all_attackers |= MoveGen::update_xrays(sq, occupied, board);

        victim = next_attacker;

        current_side = !current_side;
    }

    while (--d > 0)
    {
        value[d - 1] = -std::max(-value[d - 1], value[d]);
    }

    return value[0];
}

template int SearchWorker::see<WHITE>(int sq, Piece target, Piece attacker, int from_sq) const;
template int SearchWorker::see<BLACK>(int sq, Piece target, Piece attacker, int from_sq) const;