#include "engine/engine.hpp"

int Engine::see(int sq, Piece target, Piece attacker, Color side, int from_sq) const
{
    int value[32];
    int d = 0;
    U64 occupied = board.get_occupancy(NO_COLOR);

    // Initialisation du score avec la valeur de la pièce capturée
    value[0] = eval::get_piece_score(target);

    // On génère le masque initial de tous les attaquants
    U64 all_attackers = MoveGen::attackers_to(sq, occupied, board);

    // On retire l'attaquant initial (celui qui fait le premier coup)
    U64 used_items = (1ULL << from_sq);
    occupied ^= used_items;

    // Mise à jour des X-rays après le premier retrait
    // (Si l'attaquant initial était une pièce glissante, on regarde qui est derrière)
    if (attacker == BISHOP || attacker == ROOK || attacker == QUEEN)
    {
        all_attackers |= MoveGen::update_xrays(sq, occupied, board);
    }

    Color current_side = !side;
    Piece next_attacker = NO_PIECE;

    while (true)
    {
        // On retire les pièces déjà utilisées du masque des attaquants
        U64 current_attackers = all_attackers & occupied;

        // Trouver le plus petit attaquant pour le camp actuel
        int from = board.get_smallest_attacker(current_attackers, current_side, next_attacker);

        if (from == -1)
            break;

        d++;
        // On accumule les gains (Capture - valeur de la pièce qu'on risque)
        value[d] = eval::get_piece_score(next_attacker) - value[d - 1];

        // Retrait de la pièce utilisée
        occupied ^= (1ULL << from);

        // Mise à jour des X-rays : si on a bougé une pièce glissante,
        // on génère les nouvelles attaques qui pourraient apparaître
        if (next_attacker == PAWN || next_attacker == BISHOP || next_attacker == ROOK || next_attacker == QUEEN)
        {
            all_attackers |= MoveGen::update_xrays(sq, occupied, board);
        }

        // Si on ne peut pas améliorer le score même en capturant, on s'arrête (Alpha-Beta local)
        if (std::max(-value[d - 1], value[d]) < 0)
            break;

        current_side = !current_side;
    }

    // Remontée Minimax pour trouver le score final stabilisé
    while (--d > 0)
    {
        value[d - 1] = -std::max(-value[d - 1], value[d]);
    }

    return value[0];
}