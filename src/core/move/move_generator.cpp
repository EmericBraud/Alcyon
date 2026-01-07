#include "core/move/move_generator.hpp"

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
void MoveGen::generate_pawn_moves(Board &board, const Color color, MoveList &list)
{
    const bitboard pawns = board.get_piece_bitboard(color, PAWN);
    const bitboard occ = board.get_occupancy(NO_COLOR);
    const bitboard enemies = board.get_occupancy((Color)!color);
    const int ep_sq = board.get_en_passant_sq();

    const int current_cr = board.get_castling_rights();
    const bool ep_flag = (ep_sq != EN_PASSANT_SQ_NONE);
    const int ep_file = ep_flag ? (ep_sq % 8) : 0;

    // --- Paramètres de direction selon la couleur ---
    const int dir = (color == WHITE) ? 8 : -8;
    const int start_rank = (color == WHITE) ? 1 : 6;
    const int promo_rank = (color == WHITE) ? 6 : 1;
    const std::array<bitboard, 64> &pawn_attacks = (color == WHITE) ? PawnAttacksWhite : PawnAttacksBlack;

    bitboard temp_pawns = pawns;
    while (temp_pawns)
    {
        const int from = pop_lsb(temp_pawns);
        const int rank = from >> 3;

        // 1. Poussées (Simple et Double)
        int to = from + dir;
        // Sécurité : on vérifie que 'to' est bien sur l'échiquier avant le test d'occupation
        if (to >= 0 && to < 64 && !(occ & (1ULL << to)))
        {
            if (rank == promo_rank)
            {
                list.push(Move{from, to, PAWN, Move::Flags::PROMOTION_MASK, NO_PIECE, current_cr, ep_flag, ep_file});
            }
            else
            {
                list.push(Move{from, to, PAWN, Move::Flags::NONE, NO_PIECE, current_cr, ep_flag, ep_file});

                // Double poussée
                int double_to = to + dir;
                if (rank == start_rank && !(occ & (1ULL << double_to)))
                {
                    list.push(Move{from, double_to, PAWN, Move::Flags::DOUBLE_PUSH, NO_PIECE, current_cr, ep_flag, ep_file});
                }
            }
        }

        // 2. Captures normales
        bitboard attacks = pawn_attacks[from] & enemies;
        while (attacks)
        {
            int target = pop_lsb(attacks);
            if (rank == promo_rank)
                list.push(Move{from, target, PAWN, Move::Flags::PROMOTION_MASK, board.get_p(target), current_cr, ep_flag, ep_file});
            else
                list.push(Move{from, target, PAWN, Move::Flags::CAPTURE, board.get_p(target), current_cr, ep_flag, ep_file});
        }

        // 3. Capture En Passant
        if (ep_flag && (pawn_attacks[from] & (1ULL << ep_sq)))
        {
            list.push(Move{from, ep_sq, PAWN, Move::Flags::EN_PASSANT_CAP, PAWN, current_cr, ep_flag, ep_file});
        }
    }
}
void MoveGen::generate_pseudo_legal_moves(Board &board, const Color color, MoveList &list)
{
    const bitboard occupied = board.get_occupancy(NO_COLOR);
    const bitboard us_occ = board.get_occupancy(color);
    const bitboard opponent_king_mask = ~board.get_piece_bitboard(!color, KING);

    // 1. Pions : Déjà dans une fonction dédiée (Excellent)
    generate_pawn_moves(board, color, list);

    // 2. Cavaliers (Sauts fixes)
    bitboard knights = board.get_piece_bitboard(color, KNIGHT);
    while (knights)
    {
        int sq = pop_lsb(knights);
        bitboard targets = KnightAttacks[sq] & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, KNIGHT, targets, board);
    }

    // 3. Sliders : Fous et Tours (Utilisent l'occupancy)
    bitboard bishops = board.get_piece_bitboard(color, BISHOP);
    while (bishops)
    {
        int sq = pop_lsb(bishops);
        bitboard targets = generate_bishop_moves(sq, occupied) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, BISHOP, targets, board);
    }

    bitboard rooks = board.get_piece_bitboard(color, ROOK);
    while (rooks)
    {
        int sq = pop_lsb(rooks);
        bitboard targets = generate_rook_moves(sq, occupied) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, ROOK, targets, board);
    }

    // 4. Dames (Combinaison des deux)
    bitboard queens = board.get_piece_bitboard(color, QUEEN);
    while (queens)
    {
        int sq = pop_lsb(queens);
        bitboard targets = (generate_rook_moves(sq, occupied) | generate_bishop_moves(sq, occupied)) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, QUEEN, targets, board);
    }

    // 5. Roi
    const int sq = board.get_eval_state().king_sq[color];
    bitboard targets = KingAttacks[sq] & ~us_occ & opponent_king_mask;
    push_moves_from_mask(list, sq, KING, targets, board);

    generate_castle_moves(board, list);
}
template <Piece P>
void generate_piece_captures(const Board &board, Color color, U64 opponent_occ, MoveList &list)
{
    // Récupération des pièces du type P pour la couleur donnée
    U64 pieces = board.get_piece_bitboard(color, P);

    // Optimisation : Si pas de pièces de ce type, on sort tout de suite
    if (!pieces)
        return;

    // Pour les pions, on doit inclure la case En Passant dans les cibles valides
    // car elle n'est pas dans opponent_occ (la case est vide)
    U64 valid_targets = opponent_occ;
    if constexpr (P == PAWN)
    {
        if (board.get_en_passant_sq() != EN_PASSANT_SQ_NONE)
        {
            valid_targets |= sq_mask(board.get_en_passant_sq());
        }
    }

    while (pieces)
    {
        const int from_sq = pop_lsb(pieces);

        // Appel direct à la version template (très rapide)
        // On filtre directement avec les cibles valides (captures + EP)
        U64 attacks = MoveGen::get_pseudo_moves_mask<P>(board, from_sq, color) & valid_targets;

        while (attacks)
        {
            const int to_sq = pop_lsb(attacks);

            Move m(from_sq, to_sq, P);

            // Optimisation : On sait déjà que c'est une capture (ou EP)
            // On peut éviter MoveGen::init_move_flags complet si on veut,
            // mais gardons-le pour la sécurité pour l'instant.
            MoveGen::init_move_flags(board, m);

            list.push(m);
        }
    }
}
void MoveGen::generate_pseudo_legal_captures(const Board &board, Color color, MoveList &list)
{
    // On définit les cibles : uniquement les pièces adverses (pas le Roi pour éviter les pseudo-moves illégaux de capture de roi)
    const Color them = (color == WHITE ? BLACK : WHITE);

    // On retire le Roi adverse des cibles valides (car on ne capture jamais le roi)
    const U64 opponent_occ = board.get_occupancy(them) & ~board.get_piece_bitboard(them, KING);

    generate_piece_captures<PAWN>(board, color, opponent_occ, list);
    generate_piece_captures<KNIGHT>(board, color, opponent_occ, list);
    generate_piece_captures<BISHOP>(board, color, opponent_occ, list);
    generate_piece_captures<ROOK>(board, color, opponent_occ, list);
    generate_piece_captures<QUEEN>(board, color, opponent_occ, list);
    generate_piece_captures<KING>(board, color, opponent_occ, list);
}

/// @brief For GUI only (doesn't need high perfs)
/// @param board
/// @param from_sq
/// @return
U64 MoveGen::get_legal_moves_mask(Board &board, int from_sq)
{
    MoveList list;
    PieceInfo info = board.get_piece_on_square(from_sq);
    if (info.second == NO_PIECE)
        return 0ULL;

    U64 target_mask_pseudo = get_pseudo_moves_mask(board, from_sq);
    U64 target_mask = 0ULL;
    while (target_mask_pseudo != 0)
    {
        int to_sq = pop_lsb(target_mask_pseudo);
        Move move(from_sq, to_sq, static_cast<Piece>(info.second));
        MoveGen::init_move_flags(board, move);
        if (board.is_move_legal(move))
            target_mask |= sq_mask(to_sq);
    }

    if (info.second == KING)
    {
        generate_castle_moves(board, list);

        for (const auto move : list)
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

bool MoveGen::is_mask_attacked(const Board &board, U64 mask, Color attacker)
{
    const Color us = (attacker == WHITE) ? BLACK : WHITE;

    const std::array<U64, BOARD_SIZE> &PawnAttacks =
        (us == WHITE) ? MoveGen::PawnAttacksWhite : MoveGen::PawnAttacksBlack;

    const U64 occ = board.get_occupancy(NO_COLOR);

    // On boucle sur chaque case du masque (généralement 2 ou 3 cases pour le roque)
    while (mask)
    {
        int sq = pop_lsb(mask);

        // 1. Attaques de Pions (Reverse)
        // On regarde si un pion adverse sur sa case de départ pourrait attaquer 'sq'
        if (PawnAttacks[sq] & board.get_piece_bitboard(attacker, PAWN))
            return true;

        // 2. Attaques de Cavaliers
        if (MoveGen::KnightAttacks[sq] & board.get_piece_bitboard(attacker, KNIGHT))
            return true;

        // 3. Attaques de Fou / Dame (Diagonales)
        U64 bishop_attacks = MoveGen::generate_bishop_moves(sq, occ);
        if (bishop_attacks & (board.get_piece_bitboard(attacker, BISHOP) | board.get_piece_bitboard(attacker, QUEEN)))
            return true;

        // 4. Attaques de Tour / Dame (Colonnes/Rangées)
        U64 rook_attacks = MoveGen::generate_rook_moves(sq, occ);
        if (rook_attacks & (board.get_piece_bitboard(attacker, ROOK) | board.get_piece_bitboard(attacker, QUEEN)))
            return true;

        // 5. Attaques de Roi (Reverse)
        if (MoveGen::KingAttacks[sq] & board.get_piece_bitboard(attacker, KING))
            return true;
    }

    return false;
}
void MoveGen::generate_castle_moves(Board &board, MoveList &list)
{
    const Color us = board.get_side_to_move();
    const uint8_t castling_rights = board.get_castling_rights();
    const bitboard occupancy = board.get_occupancy(NO_COLOR);

    // On prépare les infos d'En Passant
    const bool ep_flag = board.get_en_passant_sq() != EN_PASSANT_SQ_NONE;
    const int ep_file = ep_flag ? board.get_en_passant_sq() % 8 : 0;

    // On identifie l'adversaire pour éviter switch_trait()
    const Color them = !us;

    if (us == WHITE)
    {
        // PETIT ROQUE BLANC (h1)
        // 1. Droits ? 2. Cases vides (f1, g1) ? 3. Sécurité (e1, f1, g1) ?
        if ((castling_rights & WHITE_KINGSIDE) && !(occupancy & 0x60ULL))
        {
            if (!is_mask_attacked(board, 0x70ULL, them)) // e1, f1, g1
            {
                list.push(Move(4, 6, KING, Move::Flags::KING_CASTLE, NO_PIECE, castling_rights, ep_flag, ep_file));
            }
        }
        // GRAND ROQUE BLANC (a1)
        // 1. Droits ? 2. Cases vides (b1, c1, d1) ? 3. Sécurité (c1, d1, e1) ?
        if ((castling_rights & WHITE_QUEENSIDE) && !(occupancy & 0x0EULL))
        {
            if (!is_mask_attacked(board, 0x1CULL, them)) // c1, d1, e1
            {
                list.push(Move(4, 2, KING, Move::Flags::QUEEN_CASTLE, NO_PIECE, castling_rights, ep_flag, ep_file));
            }
        }
    }
    else // BLACK
    {
        // PETIT ROQUE NOIR (h8)
        if ((castling_rights & BLACK_KINGSIDE) && !(occupancy & 0x6000000000000000ULL))
        {
            if (!is_mask_attacked(board, 0x7000000000000000ULL, them)) // e8, f8, g8
            {
                list.push(Move(60, 62, KING, Move::Flags::KING_CASTLE, NO_PIECE, castling_rights, ep_flag, ep_file));
            }
        }
        // GRAND ROQUE NOIR (a8)
        if ((castling_rights & BLACK_QUEENSIDE) && !(occupancy & 0x0E00000000000000ULL))
        {
            if (!is_mask_attacked(board, 0x1C00000000000000ULL, them)) // c8, d8, e8
            {
                list.push(Move(60, 58, KING, Move::Flags::QUEEN_CASTLE, NO_PIECE, castling_rights, ep_flag, ep_file));
            }
        }
    }
}

void MoveGen::generate_legal_moves(Board &board, MoveList &list)
{
    const Color us = board.get_side_to_move();
    MoveGen::generate_pseudo_legal_moves(board, us, list);

    const bool is_king_atck = board.is_king_attacked(us);
    bitboard king_bb = board.get_piece_bitboard(us, KING);
    const int king_sq = pop_lsb(king_bb);

    const U64 occ = board.get_occupancy(NO_COLOR);

    const U64 king_mask = generate_rook_moves(king_sq, occ) | generate_bishop_moves(king_sq, occ);
    const U64 king_mask_knight = KnightAttacks[king_sq];

    int write_idx = 0; // Pointeur d'écriture

    // On parcourt toute la liste avec un index de lecture
    for (int read_idx = 0; read_idx < list.count; ++read_idx)
    {
        Move &move = list.moves[read_idx];
        bool legal = true;

        const U64 from_msk = 1ULL << move.get_from_sq();
        const U64 to_msk = 1ULL << move.get_to_sq();

        if (move.get_from_piece() == KING || move.get_flags() == Move::Flags::EN_PASSANT_CAP)
        {
            legal = board.is_move_legal(move);
        }
        else if (!is_king_atck)
        {
            if (from_msk & king_mask)
            {
                legal = board.is_move_legal(move);
            }
        }
        else // On est en échec
        {
            if (to_msk & (king_mask | king_mask_knight))
            {
                legal = board.is_move_legal(move);
            }
            else
            {
                legal = false;
            }
        }

        // --- LE FILTRAGE ---
        if (legal)
        {
            // Si le coup est légal, on le place à la position write_idx
            // et on incrémente write_idx
            list.moves[write_idx] = move;
            list.scores[write_idx] = list.scores[read_idx];
            write_idx++;
        }
    }

    // On met à jour la taille finale de la liste
    list.count = write_idx;
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

U64 MoveGen::update_xrays(int sq, U64 occupied, const Board &board)
{
    return (generate_bishop_moves(sq, occupied) &
            (board.get_piece_bitboard(WHITE, BISHOP) | board.get_piece_bitboard(BLACK, BISHOP) |
             board.get_piece_bitboard(WHITE, QUEEN) | board.get_piece_bitboard(BLACK, QUEEN))) |
           (generate_rook_moves(sq, occupied) &
            (board.get_piece_bitboard(WHITE, ROOK) | board.get_piece_bitboard(BLACK, ROOK) |
             board.get_piece_bitboard(WHITE, QUEEN) | board.get_piece_bitboard(BLACK, QUEEN)));
}
U64 MoveGen::attackers_to(int sq, U64 occupancy, const Board &b)
{
    return ((MoveGen::PawnAttacksBlack[sq] &
             b.get_piece_bitboard(WHITE, PAWN)) |

            (MoveGen::PawnAttacksWhite[sq] &
             b.get_piece_bitboard(BLACK, PAWN)) |

            (MoveGen::KnightAttacks[sq] &
             (b.get_piece_bitboard(WHITE, KNIGHT) |
              b.get_piece_bitboard(BLACK, KNIGHT))) |

            (generate_bishop_moves(sq, occupancy) &
             (b.get_piece_bitboard(WHITE, BISHOP) |
              b.get_piece_bitboard(BLACK, BISHOP) |
              b.get_piece_bitboard(WHITE, QUEEN) |
              b.get_piece_bitboard(BLACK, QUEEN))) |

            (generate_rook_moves(sq, occupancy) &
             (b.get_piece_bitboard(WHITE, ROOK) |
              b.get_piece_bitboard(BLACK, ROOK) |
              b.get_piece_bitboard(WHITE, QUEEN) |
              b.get_piece_bitboard(BLACK, QUEEN))) |

            (MoveGen::KingAttacks[sq] &
             (b.get_piece_bitboard(WHITE, KING) |
              b.get_piece_bitboard(BLACK, KING)))) &
           occupancy;
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