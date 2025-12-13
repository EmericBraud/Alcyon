#include "move_generator.hpp"

constexpr const char *ROOK_ATTACKS_FILE = DATA_PATH "rook_attacks.bin";
constexpr const char *BISHOP_ATTACKS_FILE = DATA_PATH "bishop_attacks.bin";
constexpr const char *ROOK_MAGICS_FILE = DATA_PATH "rook_m.bin";
constexpr const char *BISHOP_MAGICS_FILE = DATA_PATH "bishop_m.bin";

namespace MoveGen
{
    std::array<U64, BOARD_SIZE> KnightAttacks;
    std::array<U64, BOARD_SIZE> KingAttacks;
    std::array<U64, BOARD_SIZE> RookMasks;
    std::array<U64, BOARD_SIZE> BishopMasks;
    std::array<U64, BOARD_SIZE> PawnAttacksWhite;
    std::array<U64, BOARD_SIZE> PawnAttacksBlack;
    std::array<U64, BOARD_SIZE> PawnPushWhite;
    std::array<U64, BOARD_SIZE> PawnPushBlack;
    std::array<U64, BOARD_SIZE> PawnPush2White;
    std::array<U64, BOARD_SIZE> PawnPush2Black;

    std::vector<U64> RookAttacksProcessing;
    std::vector<U64> BishopAttacksProcessing;

    std::array<Magic, BOARD_SIZE> RookMagics;
    std::array<Magic, BOARD_SIZE> BishopMagics;

    std::array<U64, ROOK_ATTACKS_SIZE> RookAttacks;
    std::array<U64, BISHOP_ATTACKS_SIZE> BishopAttacks;

}
/**
 * Génère le Bitboard d'attaque pour une pièce non-glissante donnée
 * sur une case donnée.
 * * @param sq La case de départ (0-63).
 * @param shifts Les déplacements possibles (tableau de {dx, dy}).
 * @return Le Bitboard d'attaque.
 */
static U64 generate_attacks(const int sq, const std::array<int, 16> &shifts)
{
    U64 attacks = 0ULL;
    const int rank = sq / 8; // Rang (ligne) 0-7
    const int file = sq % 8; // Colonne (file) 0-7

    // Boucle à travers tous les déplacements possibles
    for (int i = 0; i < static_cast<int>(shifts.size() / 2); ++i)
    {
        int target_rank = rank + shifts[i * 2];
        int target_file = file + shifts[i * 2 + 1];

        // Vérification des bords de l'échiquier
        if (target_rank >= 0 && target_rank <= 7 &&
            target_file >= 0 && target_file <= 7)
        {
            int target_sq = target_rank * 8 + target_file;

            attacks |= sq_mask(target_sq);
        }
    }
    return attacks;
}

void MoveGen::initialize_rook_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int file = sq_file + 1; file < 7; ++file)
            mask |= sq_mask(sq_rank * 8 + file);

        for (int file = sq_file - 1; file > 0; --file)
            mask |= sq_mask(sq_rank * 8 + file);

        for (int rank = sq_rank + 1; rank < 7; ++rank)
            mask |= sq_mask(rank * 8 + sq_file);

        for (int rank = sq_rank - 1; rank > 0; --rank)
            mask |= sq_mask(rank * 8 + sq_file);

        RookMasks[sq] = mask;
    }
}

void MoveGen::initialize_bishop_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int rank = sq_rank + 1, file = sq_file + 1; rank < 7 && file < 7; ++rank, ++file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file + 1; rank > 0 && file < 7; --rank, ++file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file - 1; rank > 0 && file > 0; --rank, --file)
            mask |= sq_mask(rank * 8 + file);

        for (int rank = sq_rank + 1, file = sq_file - 1; rank < 7 && file > 0; ++rank, --file)
            mask |= sq_mask(rank * 8 + file);

        BishopMasks[sq] = mask;
    }
}
void MoveGen::initialize_pawn_masks()
{
    for (int sq{0}; sq < BOARD_SIZE; ++sq)
    {
        const int sq_col = sq % 8;
        const int sq_row = sq / 8;

        // White side
        U64 white_attack_mask{0ULL};
        U64 white_push_mask{0ULL};
        U64 white_push2_mask{0ULL};
        if (sq_row < 7)
        {
            if (sq_col > 0)
            {
                white_attack_mask |= sq_mask(sq_col - 1, sq_row + 1);
            }
            if (sq_col < 7)
            {
                white_attack_mask |= sq_mask(sq_col + 1, sq_row + 1);
            }
            white_push_mask |= sq_mask(sq_col, sq_row + 1);
            if (sq_row == 1)
                white_push2_mask |= sq_mask(sq_col, sq_row + 2);
        }

        // Black side
        U64 black_attack_mask{0ULL};
        U64 black_push_mask{0ULL};
        U64 black_push2_mask{0ULL};
        if (sq_row > 0)
        {
            if (sq_col > 0)
            {
                black_attack_mask |= sq_mask(sq_col - 1, sq_row - 1);
            }
            if (sq_col < 7)
            {
                black_attack_mask |= sq_mask(sq_col + 1, sq_row - 1);
            }
            black_push_mask |= sq_mask(sq_col, sq_row - 1);
            if (sq_row == 6)
                black_push2_mask |= sq_mask(sq_col, sq_row - 2);
        }
        PawnAttacksWhite[sq] = white_attack_mask;
        PawnPushWhite[sq] = white_push_mask;
        PawnPush2White[sq] = white_push2_mask;

        PawnAttacksBlack[sq] = black_attack_mask;
        PawnPushBlack[sq] = black_push_mask;
        PawnPush2Black[sq] = black_push2_mask;
    }
}

void MoveGen::initialize_bitboard_tables()
{
    // Knight moves (8 moves : {dr, df, dr, df, ...})
    const std::array<int, 16> KNIGHT_SHIFTS = {
        -2, -1, -2, 1, // Up
        -1, -2, -1, 2, // Sides
        1, -2, 1, 2,   // Sides
        2, -1, 2, 1    // Down
    };

    // King moves (8 moves : {dr, df, dr, df, ...})
    const std::array<int, 16> KING_SHIFTS = {
        -1, 0, -1, 1,
        0, 1, 1, 1,
        1, 0, 1, -1,
        0, -1, -1, -1};

    for (int sq = 0; sq < BOARD_SIZE; ++sq)
    {
        KnightAttacks[sq] = generate_attacks(sq, KNIGHT_SHIFTS);

        KingAttacks[sq] = generate_attacks(sq, KING_SHIFTS);
    }

    MoveGen::initialize_rook_masks();
    MoveGen::initialize_bishop_masks();
    MoveGen::initialize_pawn_masks();

    std::cout << "Bitboard tables initialized." << std::endl;
}
inline U64 MoveGen::get_pseudo_moves_mask(const Board &board, const int sq, const Color color, const Piece piece_type)
{
    U64 target_mask;
    U64 temp_mask;
    const uint8_t en_passant_sq = board.get_en_passant_sq();

    switch (piece_type)
    {
    case PAWN:
        if (color == WHITE)
        {
            // We use rook move generation to limit push
            temp_mask = generate_rook_moves(sq, board) & (~board.get_occupancy(NO_COLOR));
            target_mask = (PawnPushWhite[sq] | PawnPush2White[sq]) & temp_mask;
            target_mask |= PawnAttacksWhite[sq] & board.get_occupancy(BLACK);

            if (en_passant_sq != EN_PASSANT_SQ_NONE)
            {
                target_mask |= PawnAttacksWhite[sq] & sq_mask(en_passant_sq);
            }
        }
        else if (color == BLACK)
        {
            // We use rook move generation to limit push
            temp_mask = generate_rook_moves(sq, board) & (~board.get_occupancy(NO_COLOR));
            target_mask = (PawnPushBlack[sq] | PawnPush2Black[sq]) & temp_mask;
            target_mask |= PawnAttacksBlack[sq] & board.get_occupancy(WHITE);

            if (en_passant_sq != EN_PASSANT_SQ_NONE)
            {
                target_mask |= PawnAttacksBlack[sq] & sq_mask(en_passant_sq);
            }
        }
        else
        {
            throw std::logic_error("Passed NONE color to generate pseudo legal moves");
        }
        break;
    case KNIGHT:
        target_mask = KnightAttacks[sq];
        break;
    case BISHOP:
        target_mask = generate_bishop_moves(sq, board);
        break;
    case ROOK:
        target_mask = generate_rook_moves(sq, board);
        break;
    case QUEEN:
        target_mask = generate_rook_moves(sq, board) | generate_bishop_moves(sq, board);
        break;
    case KING:
        target_mask = KingAttacks[sq];
        break;
    default:
        throw std::logic_error("Unsupported piece type");
        break;
    }

    U64 friendly_mask = board.get_occupancy(color);
    target_mask &= ~friendly_mask;
    return target_mask;
}
U64 MoveGen::get_pseudo_moves_mask(const Board &board, const int sq)
{
    const PieceInfo pair = board.get_piece_on_square(sq);
    const Color color = pair.first;
    const Piece piece_type = pair.second;
    if (piece_type == NO_PIECE)
    {
        return 0ULL;
    }
    return get_pseudo_moves_mask(board, sq, color, piece_type);
}
std::vector<Move> MoveGen::generate_pseudo_legal_moves(const Board &board, const Color color)
{
    std::vector<Move> moves;

    for (int piece_type = PAWN; piece_type <= KING; ++piece_type)
    {
        U64 piece_bitboard = board.get_piece_bitboard(color, piece_type);

        while (piece_bitboard != 0)
        {
            int from_sq = pop_lsb(piece_bitboard);
            U64 target_mask = get_pseudo_moves_mask(board, from_sq, color, static_cast<Piece>(piece_type));

            while (target_mask != 0)
            {
                int target_sq = pop_lsb(target_mask);
                moves.push_back(Move{
                    from_sq,
                    target_sq,
                    static_cast<Piece>(piece_type)});
            }
        }
    }
    return moves;
}

U64 MoveGen::get_legal_moves_mask(Board &board, int from_sq)
{
    PieceInfo info = board.get_piece_on_square(from_sq);
    if (info.second == NO_PIECE)
    {
        return 0ULL;
    }
    U64 target_mask_pseudo = get_pseudo_moves_mask(board, from_sq, info.first, static_cast<Piece>(info.second));
    U64 target_mask = 0ULL;
    while (target_mask_pseudo != 0)
    {
        int to_sq = pop_lsb(target_mask_pseudo);
        Move move(from_sq, to_sq, info.second);
        board.play(move);
        if (!is_king_attacked(board))
        {
            target_mask |= sq_mask(to_sq);
        }
        board.unplay(move);
    }

    if (info.second == KING)
    {
        std::vector<Move> moves = generate_castle_moves(board);

        for (const auto move : moves)
        {
            if (move.get_flags() == Move::Flags::KING_CASTLE)
            {
                const U64 mask = info.first == WHITE ? (1ULL << 6) : (1ULL << 62);
                target_mask |= mask;
            }
            if (move.get_flags() == Move::Flags::QUEEN_CASTLE)
            {
                const U64 mask = info.first == WHITE ? (1ULL << 2) : (1ULL << 58);
                target_mask |= mask;
            }
        }
    }

    return target_mask;
}

bool MoveGen::is_king_attacked(Board &board)
{
    const Color color = (board.get_side_to_move() == WHITE ? BLACK : WHITE);
    const Color opponent_color = (color == WHITE) ? BLACK : WHITE;
    bitboard king_mask = board.get_piece_bitboard(color, KING);
    assert(king_mask);
    bitboard king_mask_cpy = king_mask;
    const int king_sq{pop_lsb(king_mask_cpy)};
    assert(king_sq >= 0 && king_sq <= 63);

    const U64 rook_check = generate_rook_moves(king_sq, board);
    if (rook_check & (board.get_piece_bitboard(opponent_color, QUEEN) | board.get_piece_bitboard(opponent_color, ROOK)))
    {
        return true;
    }
    const U64 bishop_check = generate_bishop_moves(king_sq, board);
    if (bishop_check & (board.get_piece_bitboard(opponent_color, QUEEN) | board.get_piece_bitboard(opponent_color, BISHOP)))
    {
        return true;
    }

    const U64 knight_check = KnightAttacks[king_sq];
    if (knight_check & board.get_piece_bitboard(opponent_color, KNIGHT))
    {
        return true;
    }

    const U64 king_check = KingAttacks[king_sq];
    if (king_check & board.get_piece_bitboard(opponent_color, KING))
    {
        return true;
    }

    const U64 pawn_check = (color == WHITE) ? PawnAttacksWhite[king_sq] : PawnAttacksBlack[king_sq];
    if (pawn_check & board.get_piece_bitboard(opponent_color, PAWN))
    {
        return true;
    }
    return false;
}
bool MoveGen::is_mask_attacked(Board &board, const U64 mask)
{
    const Color color = board.get_side_to_move();
    for (int piece_type = PAWN; piece_type <= KING; ++piece_type)
    {
        U64 piece_bitboard = board.get_piece_bitboard(color, piece_type);
        while (piece_bitboard != 0)
        {
            int from_sq = pop_lsb(piece_bitboard);
            U64 target_mask = get_pseudo_moves_mask(board, from_sq, color, static_cast<Piece>(piece_type));
            if (mask & target_mask)
            {
                return true;
            }
        }
    }
    return false;
}
std::vector<Move> MoveGen::generate_castle_moves(Board &board)
{
    std::vector<Move> moves;
    moves.reserve(2);
    const Color color = board.get_side_to_move();
    const uint8_t castling_rights = board.get_castling_rights();
    const bitboard occupancy = board.get_occupancy(NO_COLOR);
    board.switch_trait();
    if (color == WHITE)
    {
        if ((castling_rights & WHITE_KINGSIDE) && !(occupancy & 0x60) && (!is_mask_attacked(board, 0x60)))
        {
            const Move move{4, 6, KING, Move::Flags::KING_CASTLE};
            moves.push_back(move);
        }
        if ((castling_rights & WHITE_QUEENSIDE) && !(occupancy & 0xE) && !(is_mask_attacked(board, 0xE)))
        {
            const Move move{4, 2, KING, Move::Flags::QUEEN_CASTLE};
            moves.push_back(move);
        }
    }
    else
    {
        if ((castling_rights & BLACK_KINGSIDE) && !(occupancy & 0x6000000000000000) && (!is_mask_attacked(board, 0x6000000000000000)))
        {
            const Move move{60, 62, KING, Move::Flags::KING_CASTLE};
            moves.push_back(move);
        }
        if ((castling_rights & BLACK_QUEENSIDE) && !(occupancy & 0xE00000000000000) && !(is_mask_attacked(board, 0xE00000000000000)))
        {
            const Move move{60, 58, KING, Move::Flags::QUEEN_CASTLE};
            moves.push_back(move);
        }
    }
    board.switch_trait();
    return moves;
}
std::vector<Move> MoveGen::generate_legal_moves(Board &board)
{
    const Color color = board.get_side_to_move();
    std::vector<Move> moves = MoveGen::generate_pseudo_legal_moves(board, color);
    for (auto iter = moves.begin(); iter != moves.end();)
    {
        Move &move = *iter;
        board.play(move);
        if (MoveGen::is_king_attacked(board))
        {
            board.unplay(move);
            iter = moves.erase(iter);
        }
        else
        {
            board.unplay(move);
            iter++;
        }
    }
    return moves;
}

U64 MoveGen::generate_rook_moves(int from_sq, const Board &board)
{
    const U64 occupancy = board.get_occupancy(NO_COLOR);
    const MoveGen::Magic magic = MoveGen::RookMagics[from_sq];

    const U64 index = (((occupancy & magic.mask) * magic.magic) >> magic.shift);

    return MoveGen::RookAttacks[index + magic.index_start];
}

U64 MoveGen::generate_bishop_moves(int from_sq, const Board &board)
{
    const U64 occupancy = board.get_occupancy(NO_COLOR);
    const MoveGen::Magic magic = MoveGen::BishopMagics[from_sq];

    const U64 index = (((occupancy & magic.mask) * magic.magic) >> magic.shift);

    return MoveGen::BishopAttacks[index + magic.index_start];
}

U64 generate_sliding_attack(int sq, U64 occupancy, bool is_rook)
{
    U64 attacks = 0ULL;

    const std::array<std::pair<int, int>, 4> dirs_rook = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};     // N,S,E,W (dr,df)
    const std::array<std::pair<int, int>, 4> dirs_bishop = {{{1, 1}, {-1, 1}, {-1, -1}, {1, -1}}}; // NE,NW,SW,SE

    const auto &dirs = is_rook ? dirs_rook : dirs_bishop;

    int r0 = sq / 8;
    int f0 = sq % 8;

    for (const auto &d : dirs)
    {
        int r = r0 + d.first;
        int f = f0 + d.second;
        while (r >= 0 && r <= 7 && f >= 0 && f <= 7)
        {
            int target_sq = r * 8 + f;
            U64 target_mask = sq_mask(target_sq);
            attacks |= target_mask;
            if (occupancy & target_mask) // bloqueur rencontré -> s'arrête
                break;
            r += d.first;
            f += d.second;
        }
    }
    return attacks;
}

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
            std::cout << "Magic found for sq " << sq << " after " << iter << " iterations." << std::endl;
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

    std::cout << "--- Exported attack table for " << (is_rook ? "rook" : "bishop") << " piece ---" << std::endl;
}

void MoveGen::run_magic_searcher()
{
    std::cout << "--- Searching Magic Numbers ---" << std::endl;
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

    std::cout << "--- Magic Numbers Exported ---" << std::endl;
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
    std::cout << (is_rook ? "Rook " : "Bishop ") << "file size : " << attacks_file_sz << std::endl;
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

static bool perform_initial_data_loading()
{
    MoveGen::initialize_bitboard_tables();

    MoveGen::load_magics();

    MoveGen::load_attacks();

    std::cout << "--- All bitboard tables loaded successfully ! ---" << std::endl;
    return true;
}

static bool __bitboard_data_loaded = perform_initial_data_loading();