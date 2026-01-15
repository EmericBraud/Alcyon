#include "engine/search/worker.hpp"

#include "core/move/generator/move_generator.hpp"

int SearchWorker::see(int sq, Piece target, Piece attacker, Color side, int from_sq) const
{
    int value[32];
    int d = 0;
    U64 occupied = board.get_occupancy(NO_COLOR);

    // 1. Gain initial
    value[0] = Eval::get_piece_score(target);

    // On stocke la prochaine victime.
    // Au prochain tour, l'adversaire mangera 'attacker'.
    Piece victim = attacker;

    U64 all_attackers = MoveGen::attackers_to(sq, occupied, board);

    occupied ^= (1ULL << from_sq);
    all_attackers |= MoveGen::update_xrays(sq, occupied, board);

    Color current_side = !side;
    Piece next_attacker = NO_PIECE;

    while (true)
    {
        d++;
        U64 potential_attackers = all_attackers & occupied;
        int from = board.get_smallest_attacker(potential_attackers, current_side, next_attacker);

        if (from == -1)
            break;

        // --- CORRECTION MAJEURE ---
        // La valeur gagnée ce tour-ci est celle de la VICTIME (celle qui occupait la case),
        // moins le score accumulé jusqu'ici.
        value[d] = Eval::get_piece_score(victim) - value[d - 1];

        if (std::max(-value[d - 1], value[d]) < 0)
            break;

        occupied ^= (1ULL << from);
        all_attackers |= MoveGen::update_xrays(sq, occupied, board);

        // Préparation du tour suivant :
        // Le nouvel attaquant devient la victime du prochain coup
        victim = next_attacker;

        current_side = !current_side;
    }

    while (--d > 0)
    {
        value[d - 1] = -std::max(-value[d - 1], value[d]);
    }

    return value[0];
}