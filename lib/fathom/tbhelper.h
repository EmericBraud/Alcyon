#ifndef TBCHESS_HELPERS_H
#define TBCHESS_HELPERS_H

#include <stdint.h>

/* version tbchess des fonctions bitboard */
static inline unsigned tbchess_lsb(uint64_t b)
{
    return __builtin_ctzll(b);
}

static inline uint64_t tbchess_poplsb(uint64_t b)
{
    return b & (b - 1) ^ b; // retire le LSB et renvoie sa valeur
}

static inline unsigned tbchess_popcount(uint64_t b)
{
    return __builtin_popcountll(b);
}

#endif
