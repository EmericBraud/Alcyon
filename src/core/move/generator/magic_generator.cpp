// This contains functions necessary to generate the magic number tables.
// This should not be used at all since magic tables are now already exported in data folder
// But let's keep the code somewhere

#ifndef __BMI2__
static constexpr const char *ROOK_ATTACKS_FILE = DATA_PATH "rook_attacks.bin";
static constexpr const char *BISHOP_ATTACKS_FILE = DATA_PATH "bishop_attacks.bin";
static constexpr const char *ROOK_MAGICS_FILE = DATA_PATH "rook_m.bin";
static constexpr const char *BISHOP_MAGICS_FILE = DATA_PATH "bishop_m.bin";

#include "move_generator.hpp"

#include <vector>

#include "common/logger.hpp"

static void generate_all_blocker_occupancies(int sq, U64 mask, bool is_rook,
                                             std::vector<U64> &occupancy_list,
                                             std::vector<U64> &attack_list)
{
    std::vector<int> blocker_bits;
    U64 temp_mask = mask;
    while (temp_mask)
    {
        blocker_bits.push_back(pop_lsb(temp_mask));
    }

    int num_blocker_bits = blocker_bits.size();

    uint64_t total = 1ULL << num_blocker_bits;
    for (uint64_t i = 0; i < total; ++i)
    {
        U64 occupancy = 0ULL;
        for (int j = 0; j < num_blocker_bits; ++j)
            if ((i >> j) & 1ULL) // Chekcs if jth blocker is present
                occupancy |= sq_mask(blocker_bits[j]);

        occupancy_list.push_back(occupancy);
        attack_list.push_back(generate_sliding_attack(sq, occupancy, is_rook));
    }
}

static std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());

static U64 generate_magic_candidate()
{
    return rng() & rng() & rng();
}
static MoveGen::Magic find_magic(int sq, bool is_rook, int max_iterations, long unsigned int index_start)
{
    U64 mask = is_rook ? MoveGen::RookMasks[sq] : MoveGen::BishopMasks[sq];

    int bits_in_mask = std::popcount(mask);
    int shift = BOARD_SIZE - bits_in_mask;

    std::vector<U64> occupancies, attacks;
    generate_all_blocker_occupancies(sq, mask, is_rook, occupancies, attacks);
    int num_occupancies = occupancies.size();

    for (int iter = 0; iter < max_iterations; ++iter)
    {
        U64 magic_candidate = generate_magic_candidate();

        if (((mask * magic_candidate) & 0xFF00000000000000ULL) == 0)
            continue;

        std::vector<U64> magic_test_table(num_occupancies, 0ULL);
        bool collision_found = false;

        for (int i = 0; i < num_occupancies; ++i)
        {
            U64 occupancy = occupancies[i];
            U64 ideal_attack = attacks[i];

            int index = (int)((occupancy * magic_candidate) >> shift);
            if (index >= num_occupancies)
            {
                collision_found = true;
                break;
            }
            if (magic_test_table[index] == 0ULL)
            {
                magic_test_table[index] = ideal_attack;
            }
            else if (magic_test_table[index] != ideal_attack)
            {
                collision_found = true;
                break;
            }
        }

        if (!collision_found)
        {
            logs::debug << "Magic found for sq " << sq << " after " << iter << " iterations." << std::endl;
            return {mask, magic_candidate, shift, index_start};
        }
    }

    throw std::runtime_error("No magic number found for square " + std::to_string(sq));
}

void MoveGen::export_attack_table(const std::array<MoveGen::Magic, BOARD_SIZE> m_array, bool is_rook)
{
    std::vector<U64> output_v{};
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const MoveGen::Magic magic = m_array[sq];
        const U64 mask = is_rook ? MoveGen::RookMasks[sq] : MoveGen::BishopMasks[sq];
        std::vector<U64> occupancies, attacks;
        generate_all_blocker_occupancies(sq, mask, is_rook, occupancies, attacks);

        if (magic.index_start != output_v.size())
        {
            throw std::logic_error("Index start and real size not matching");
        }
        output_v.insert(output_v.end(), static_cast<size_t>(1ULL << (BOARD_SIZE - magic.shift)), 0ULL);
        for (long unsigned int i{0}; i < occupancies.size(); ++i)
        {
            U64 index = (occupancies[i] * magic.magic) >> magic.shift;
            output_v[magic.index_start + index] = attacks[i];
        }
    }
    std::ofstream piece_attacks(is_rook ? ROOK_ATTACKS_FILE : BISHOP_ATTACKS_FILE, std::ios::binary);
    if (!piece_attacks.is_open())
    {
        throw std::runtime_error("Could not write in attacks file");
    }
    piece_attacks.write(reinterpret_cast<const char *>(output_v.data()), output_v.size() * sizeof(U64));

    logs::debug << "--- Exported attack table for " << (is_rook ? "rook" : "bishop") << " piece ---" << std::endl;
}

void MoveGen::run_magic_searcher()
{
    logs::debug << "--- Searching Magic Numbers ---" << std::endl;
    MoveGen::initialize_rook_masks();
    MoveGen::initialize_bishop_masks();

    std::array<MoveGen::Magic, BOARD_SIZE> rook_m_array;
    std::array<MoveGen::Magic, BOARD_SIZE> bishop_m_array;

    long unsigned int index_start_bishop = 0;
    long unsigned int index_start_rook = 0;

    for (int sq = 0; sq < BOARD_SIZE; ++sq)
    {
        MoveGen::Magic rook_m = find_magic(sq, true, 10000000, index_start_rook);
        MoveGen::Magic bishop_m = find_magic(sq, false, 10000000, index_start_bishop);

        if (rook_m.shift == 0 || bishop_m.shift == 0) // search failed
        {
            std::cerr << "Error: Magic search failed for square " << sq << std::endl;
            return;
        }
        index_start_bishop += 1ULL << (BOARD_SIZE - bishop_m.shift);
        index_start_rook += 1ULL << (BOARD_SIZE - rook_m.shift);
        rook_m_array[sq] = rook_m;
        bishop_m_array[sq] = bishop_m;
    }
    std::ofstream rook_m_file(ROOK_MAGICS_FILE, std::ios::binary);
    std::ofstream bishop_m_file(BISHOP_MAGICS_FILE, std::ios::binary);
    if (!rook_m_file.is_open() || !bishop_m_file.is_open())
    {
        throw std::runtime_error("Magic number files could not be opened");
    }

    rook_m_file.write(
        reinterpret_cast<const char *>(rook_m_array.data()),
        rook_m_array.size() * sizeof(MoveGen::Magic));

    bishop_m_file.write(
        reinterpret_cast<const char *>(bishop_m_array.data()),
        bishop_m_array.size() * sizeof(MoveGen::Magic));

    logs::debug << "--- Magic Numbers Exported ---" << std::endl;
    MoveGen::export_attack_table(rook_m_array, true);
    MoveGen::export_attack_table(bishop_m_array, false);
}

void MoveGen::get_sizes(bool is_rook)
{
    std::ifstream attacks_file((is_rook ? ROOK_ATTACKS_FILE : BISHOP_ATTACKS_FILE), std::ios::binary | std::ios::ate);

    if (!attacks_file.is_open())
    {
        throw std::runtime_error("File can't be opened to check size");
    }
    std::streamsize attacks_file_sz = attacks_file.tellg();

    if (attacks_file_sz % sizeof(U64) != 0)
    {
        throw std::runtime_error("Attack file size isn't a mutliple of U64 size");
    }
    logs::debug << (is_rook ? "Rook " : "Bishop ") << "file size : " << attacks_file_sz << std::endl;
}
void MoveGen::load_magics(bool is_rook)
{
    std::ifstream file((is_rook ? ROOK_MAGICS_FILE : BISHOP_MAGICS_FILE), std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Magics file couln't be opened for read operation");
    }
    file.read(reinterpret_cast<char *>(
                  (is_rook ? MoveGen::RookMagics : MoveGen::BishopMagics).data()),
              BOARD_SIZE * sizeof(MoveGen::Magic));
}
void MoveGen::load_magics()
{
    load_magics(true);
    load_magics(false);
}

void MoveGen::load_attacks_rook()
{
    std::ifstream file(ROOK_ATTACKS_FILE, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Attacks file couln't be opened for read operation");
    }
    file.read(reinterpret_cast<char *>(
                  MoveGen::RookAttacks.data()),
              ROOK_ATTACKS_SIZE * sizeof(U64));
    if (file.gcount() != ROOK_ATTACKS_SIZE * sizeof(U64))
    {
        throw std::runtime_error("Failed to read complete rook attacks table");
    }
}
void MoveGen::load_attacks_bishop()
{
    std::ifstream file(BISHOP_ATTACKS_FILE, std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Attacks file couln't be opened for read operation");
    }
    file.read(reinterpret_cast<char *>(
                  MoveGen::BishopAttacks.data()),
              BISHOP_ATTACKS_SIZE * sizeof(U64));
    if (file.gcount() != BISHOP_ATTACKS_SIZE * sizeof(U64))
    {
        throw std::runtime_error("Failed to read complete bishop attacks table");
    }
}

void MoveGen::load_attacks()
{
    load_attacks_bishop();
    load_attacks_rook();
}

#endif