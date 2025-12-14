#pragma once

#include "transp_table.hpp"
#define HASH_PIECE(color, piece, sq) (zobrist_key ^= zobrist_table[(piece) + ((color) == BLACK ? 6 : 0)][(sq)])

extern uint64_t zobrist_table[12][64];
extern uint64_t zobrist_castling[16];
extern uint64_t zobrist_en_passant[9];
extern uint64_t zobrist_black_to_move;

void init_zobrist();
