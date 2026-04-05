// This file contains code that should only be used when BMI2 instructions are available.
// Its goal is to generate PEXT attack tables and store them in separate files in the data folder.
// This code MUST NOT be executed during normal engine runtime.

#ifdef __BMI2__

#include "move_generator.hpp"

#include <bit>
#include <vector>
#include <fstream>
#include <bit>
#include <stdexcept>

#include "common/file.hpp"

// -----------------------------------------------------------------------------
// Helper: generate all PEXT attacks for one square
// -----------------------------------------------------------------------------
static void generate_pext_attacks_for_square(
    int sq,
    U64 mask,
    bool is_rook,
    std::vector<U64> &attacks)
{
    const int bits = std::popcount(mask);
    const int combinations = 1 << bits;

    int bit_positions[16];
    int count = 0;
    for (int i = 0; i < 64; ++i)
        if (mask & (1ULL << i))
            bit_positions[count++] = i;

    for (int i = 0; i < combinations; ++i)
    {
        U64 occ = 0ULL;
        for (int b = 0; b < bits; ++b)
            if (i & (1 << b))
                occ |= 1ULL << bit_positions[b];

        attacks.push_back(generate_sliding_attack(sq, occ, is_rook));
    }
}

// -----------------------------------------------------------------------------
// Export tables (one-shot tool)
// -----------------------------------------------------------------------------
void MoveGen::export_attack_tables()
{
    initialize_rook_masks();
    initialize_bishop_masks();

    std::vector<U64> rook_attacks;
    std::vector<U64> bishop_attacks;

    // Generate ROOK tables
    for (int sq = 0; sq < constants::BoardSize; ++sq)
    {
        MagicPEXT &m = RookMagics[sq];
        m.mask = RookMasks[sq];
        m.index_start = rook_attacks.size();

        generate_pext_attacks_for_square(
            sq, m.mask, true, rook_attacks);
    }

    // Generate BISHOP tables
    for (int sq = 0; sq < constants::BoardSize; ++sq)
    {
        MagicPEXT &m = BishopMagics[sq];
        m.mask = BishopMasks[sq];
        m.index_start = bishop_attacks.size();

        generate_pext_attacks_for_square(
            sq, m.mask, false, bishop_attacks);
    }

    // Write attack tables
    {
        std::ofstream out(file::get_data_path("rook_attacks_pext.bin"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(rook_attacks.data()),
                  rook_attacks.size() * sizeof(U64));
    }
    {
        std::ofstream out(file::get_data_path("bishop_attacks_pext.bin"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(bishop_attacks.data()),
                  bishop_attacks.size() * sizeof(U64));
    }

    // Write MagicPEXT tables
    {
        std::ofstream out(file::get_data_path("rook_pext.bin"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(RookMagics.data()),
                  constants::BoardSize * sizeof(MagicPEXT));
    }
    {
        std::ofstream out(file::get_data_path("bishop_pext.bin"), std::ios::binary);
        out.write(reinterpret_cast<const char *>(BishopMagics.data()),
                  constants::BoardSize * sizeof(MagicPEXT));
    }
}

// -----------------------------------------------------------------------------
// Load tables (runtime, BMI2 only)
// -----------------------------------------------------------------------------
void MoveGen::load_attack_tables()
{
    // Load rook attacks
    {
        std::ifstream in(file::get_data_path("rook_attacks_pext.bin"), std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open rook_attacks_pext.bin");

        in.read(reinterpret_cast<char *>(RookAttacks.data()),
                RookAttacks.size() * sizeof(U64));
    }

    // Load bishop attacks
    {
        std::ifstream in(file::get_data_path("bishop_attacks_pext.bin"), std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open bishop_attacks_pext.bin");

        in.read(reinterpret_cast<char *>(BishopAttacks.data()),
                BishopAttacks.size() * sizeof(U64));
    }

    // Load MagicPEXT info
    {
        std::ifstream in(file::get_data_path("rook_pext.bin"), std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open rook_pext.bin");

        in.read(reinterpret_cast<char *>(RookMagics.data()),
                constants::BoardSize * sizeof(MagicPEXT));
    }
    {
        std::ifstream in(file::get_data_path("bishop_pext.bin"), std::ios::binary);
        if (!in)
            throw std::runtime_error("Failed to open bishop_pext.bin");

        in.read(reinterpret_cast<char *>(BishopMagics.data()),
                constants::BoardSize * sizeof(MagicPEXT));
    }
}

#endif
