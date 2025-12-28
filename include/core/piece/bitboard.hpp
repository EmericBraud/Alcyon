#pragma once
#include "utils.hpp"

#define EMPTY_MASK 0ULL

inline bool is_set(const bitboard b, const int sq)
{
    return b & (1ULL << sq);
}

inline U64 sq_mask(const int sq)
{
    return (1ULL << sq);
}
inline U64 sq_mask(const int sq_col, const int sq_row)
{
    return (1ULL << (sq_col + (sq_row << 3)));
}