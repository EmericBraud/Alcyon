#include "zobrist.hpp"

uint64_t zobrist_table[12][64];
uint64_t zobrist_castling[16];
uint64_t zobrist_en_passant[9];
uint64_t zobrist_black_to_move;

void init_zobrist()
{
    std::mt19937_64 gen(12345);
    std::uniform_int_distribution<uint64_t> dist;

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
    zobrist_black_to_move = dist(gen);
}
