#pragma once
#include <cstdint>

using U64 = std::uint64_t;

namespace core::mask
{
    constexpr inline bool is_set(const U64 b, const int sq)
    {
        return b & (1ULL << sq);
    }

    constexpr inline U64 sq_mask(const int sq)
    {
        return (1ULL << sq);
    }
    constexpr inline U64 sq_mask(const int sq_col, const int sq_row)
    {
        return (1ULL << (sq_col + (sq_row << 3)));
    }

    constexpr U64 EmptyMask = 0ULL;

    static constexpr uint8_t KingVicinityFile[8] = {
        0b00000011, // File A (0) : colonnes A et B
        0b00000111, // File B (1) : colonnes A, B, C
        0b00001110, // File C (2) : colonnes B, C, D
        0b00011100, // File D (3) : colonnes C, D, E
        0b00111000, // File E (4) : colonnes D, E, F
        0b01110000, // File F (5) : colonnes E, F, G
        0b11100000, // File G (6) : colonnes F, G, H
        0b11000000  // File H (7) : colonnes G et H
    };
    static constexpr uint64_t File[8] = {
        0x0101010101010101ULL, // Colonne A
        0x0202020202020202ULL, // Colonne B
        0x0404040404040404ULL, // Colonne C
        0x0808080808080808ULL, // Colonne D
        0x1010101010101010ULL, // Colonne E
        0x2020202020202020ULL, // Colonne F
        0x4040404040404040ULL, // Colonne G
        0x8080808080808080ULL  // Colonne H
    };

}
