#include "engine/zobrist.hpp"

#include <random>

U64 zobrist_table[12][64];
U64 zobrist_castling[16];
U64 zobrist_en_passant[9];
U64 zobrist_side_to_move;

void init_zobrist()
{
    std::mt19937_64 gen(12345);
    std::uniform_int_distribution<U64> dist;

    for (int p = 0; p < 12; ++p)
    {
        for (int sq = 0; sq < 64; ++sq)
        {
            zobrist_table[p][sq] = dist(gen);
        }
    }

    for (int i = 0; i < 16; ++i)
        zobrist_castling[i] = dist(gen);
    for (int i = 0; i < 9; ++i)
        zobrist_en_passant[i] = dist(gen);
    zobrist_side_to_move = dist(gen);
}
