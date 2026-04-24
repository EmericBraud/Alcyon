// This file contains code that should only be used when BMI2 instructions are available.
// Its goal is to generate PEXT attack tables and store them in separate files in the data folder.
// This code MUST NOT be executed during normal engine runtime.

#ifdef __BMI2__

#include "move_generator.hpp"

#include <bit>
#include <vector>
#include <fstream>
#include <bit>

#include "common/file.hpp"
#include "common/logger.hpp"
#include "common/fatal.hpp"

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

static void generate_tables_in_memory()
{
    MoveGen::initialize_rook_masks();
    MoveGen::initialize_bishop_masks();

    std::size_t rook_index = 0;
    for (int sq = 0; sq < constants::BoardSize; ++sq)
    {
        MoveGen::MagicPEXT &m = MoveGen::RookMagics[sq];
        m.mask = MoveGen::RookMasks[sq];
        m.index_start = rook_index;

        std::vector<U64> attacks;
        attacks.reserve(1ULL << std::popcount(m.mask));
        generate_pext_attacks_for_square(sq, m.mask, true, attacks);

        for (std::size_t i = 0; i < attacks.size(); ++i)
            MoveGen::RookAttacks[rook_index + i] = attacks[i];

        rook_index += attacks.size();
    }

    std::size_t bishop_index = 0;
    for (int sq = 0; sq < constants::BoardSize; ++sq)
    {
        MoveGen::MagicPEXT &m = MoveGen::BishopMagics[sq];
        m.mask = MoveGen::BishopMasks[sq];
        m.index_start = bishop_index;

        std::vector<U64> attacks;
        attacks.reserve(1ULL << std::popcount(m.mask));
        generate_pext_attacks_for_square(sq, m.mask, false, attacks);

        for (std::size_t i = 0; i < attacks.size(); ++i)
            MoveGen::BishopAttacks[bishop_index + i] = attacks[i];

        bishop_index += attacks.size();
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
    std::ifstream rook_attacks_in(file::get_data_path("rook_attacks_pext.bin"), std::ios::binary);
    std::ifstream bishop_attacks_in(file::get_data_path("bishop_attacks_pext.bin"), std::ios::binary);
    std::ifstream rook_pext_in(file::get_data_path("rook_pext.bin"), std::ios::binary);
    std::ifstream bishop_pext_in(file::get_data_path("bishop_pext.bin"), std::ios::binary);

    if (!rook_attacks_in || !bishop_attacks_in || !rook_pext_in || !bishop_pext_in)
    {
        logs::error << "PEXT data files missing, generating attack tables in memory" << std::endl;
        generate_tables_in_memory();
        return;
    }

    // Load rook attacks
    {
        rook_attacks_in.read(reinterpret_cast<char *>(RookAttacks.data()),
                             RookAttacks.size() * sizeof(U64));
        if (rook_attacks_in.gcount() != static_cast<std::streamsize>(RookAttacks.size() * sizeof(U64)))
            FATAL("Failed to read rook_attacks_pext.bin");
    }

    // Load bishop attacks
    {
        bishop_attacks_in.read(reinterpret_cast<char *>(BishopAttacks.data()),
                               BishopAttacks.size() * sizeof(U64));
        if (bishop_attacks_in.gcount() != static_cast<std::streamsize>(BishopAttacks.size() * sizeof(U64)))
            FATAL("Failed to read bishop_attacks_pext.bin");
    }

    // Load MagicPEXT info
    {
        rook_pext_in.read(reinterpret_cast<char *>(RookMagics.data()),
                          constants::BoardSize * sizeof(MagicPEXT));
        if (rook_pext_in.gcount() != static_cast<std::streamsize>(constants::BoardSize * sizeof(MagicPEXT)))
            FATAL("Failed to read rook_pext.bin");
    }
    {
        bishop_pext_in.read(reinterpret_cast<char *>(BishopMagics.data()),
                            constants::BoardSize * sizeof(MagicPEXT));
        if (bishop_pext_in.gcount() != static_cast<std::streamsize>(constants::BoardSize * sizeof(MagicPEXT)))
            FATAL("Failed to read bishop_pext.bin");
    }
}

#endif
