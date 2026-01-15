#include "move_generator.hpp"

#include "common/logger.hpp"
namespace MoveGen
{
    alignas(64) std::array<U64, constants::BoardSize> KnightAttacks;
    alignas(64) std::array<U64, constants::BoardSize> KingAttacks;
    alignas(64) std::array<U64, constants::BoardSize> RookMasks;
    alignas(64) std::array<U64, constants::BoardSize> BishopMasks;
    alignas(64) std::array<U64, constants::BoardSize> PawnAttacksWhite;
    alignas(64) std::array<U64, constants::BoardSize> PawnAttacksBlack;
    alignas(64) std::array<U64, constants::BoardSize> PawnPushWhite;
    alignas(64) std::array<U64, constants::BoardSize> PawnPushBlack;
    alignas(64) std::array<U64, constants::BoardSize> PawnPush2White;
    alignas(64) std::array<U64, constants::BoardSize> PawnPush2Black;

#ifdef __BMI2__
    alignas(64) std::array<MagicPEXT, constants::BoardSize> RookMagics;
    std::array<MagicPEXT, constants::BoardSize> BishopMagics;
#else
    alignas(64) std::array<Magic, constants::BoardSize> RookMagics;
    alignas(64) std::array<Magic, constants::BoardSize> BishopMagics;
#endif

    alignas(64) std::array<U64, file::RookAttacksFileSize> RookAttacks;
    alignas(64) std::array<U64, file::BishopAttacksFileSize> BishopAttacks;

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

            attacks |= core::mask::sq_mask(target_sq);
        }
    }
    return attacks;
}

void MoveGen::initialize_rook_masks()
{
    for (int sq{0}; sq < constants::BoardSize; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int file = sq_file + 1; file < 7; ++file)
            mask |= core::mask::sq_mask(sq_rank * 8 + file);

        for (int file = sq_file - 1; file > 0; --file)
            mask |= core::mask::sq_mask(sq_rank * 8 + file);

        for (int rank = sq_rank + 1; rank < 7; ++rank)
            mask |= core::mask::sq_mask(rank * 8 + sq_file);

        for (int rank = sq_rank - 1; rank > 0; --rank)
            mask |= core::mask::sq_mask(rank * 8 + sq_file);

        RookMasks[sq] = mask;
    }
}

void MoveGen::initialize_bishop_masks()
{
    for (int sq{0}; sq < constants::BoardSize; ++sq)
    {
        const int sq_rank = sq / 8;
        const int sq_file = sq % 8;
        U64 mask = 0ULL;

        for (int rank = sq_rank + 1, file = sq_file + 1; rank < 7 && file < 7; ++rank, ++file)
            mask |= core::mask::sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file + 1; rank > 0 && file < 7; --rank, ++file)
            mask |= core::mask::sq_mask(rank * 8 + file);

        for (int rank = sq_rank - 1, file = sq_file - 1; rank > 0 && file > 0; --rank, --file)
            mask |= core::mask::sq_mask(rank * 8 + file);

        for (int rank = sq_rank + 1, file = sq_file - 1; rank < 7 && file > 0; ++rank, --file)
            mask |= core::mask::sq_mask(rank * 8 + file);

        BishopMasks[sq] = mask;
    }
}
void MoveGen::initialize_pawn_masks()
{
    for (int sq{0}; sq < constants::BoardSize; ++sq)
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
                white_attack_mask |= core::mask::sq_mask(sq_col - 1, sq_row + 1);
            }
            if (sq_col < 7)
            {
                white_attack_mask |= core::mask::sq_mask(sq_col + 1, sq_row + 1);
            }
            white_push_mask |= core::mask::sq_mask(sq_col, sq_row + 1);
            if (sq_row == 1)
                white_push2_mask |= core::mask::sq_mask(sq_col, sq_row + 2);
        }

        // Black side
        U64 black_attack_mask{0ULL};
        U64 black_push_mask{0ULL};
        U64 black_push2_mask{0ULL};
        if (sq_row > 0)
        {
            if (sq_col > 0)
            {
                black_attack_mask |= core::mask::sq_mask(sq_col - 1, sq_row - 1);
            }
            if (sq_col < 7)
            {
                black_attack_mask |= core::mask::sq_mask(sq_col + 1, sq_row - 1);
            }
            black_push_mask |= core::mask::sq_mask(sq_col, sq_row - 1);
            if (sq_row == 6)
                black_push2_mask |= core::mask::sq_mask(sq_col, sq_row - 2);
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

    for (int sq = 0; sq < constants::BoardSize; ++sq)
    {
        KnightAttacks[sq] = generate_attacks(sq, KNIGHT_SHIFTS);

        KingAttacks[sq] = generate_attacks(sq, KING_SHIFTS);
    }

    MoveGen::initialize_rook_masks();
    MoveGen::initialize_bishop_masks();
    MoveGen::initialize_pawn_masks();

    logs::debug << "Bitboard tables initialized." << std::endl;
}
void MoveGen::generate_pawn_moves(Board &board, const Color color, MoveList &list)
{
    const U64 pawns = board.get_piece_bitboard(color, PAWN);
    const U64 occ = board.get_occupancy(NO_COLOR);
    const U64 enemies = board.get_occupancy((Color)!color);
    const int ep_sq = board.get_en_passant_sq();

    const bool ep_flag = (ep_sq != constants::EnPassantSqNone);

    // --- Paramètres de direction selon la couleur ---
    const int dir = (color == WHITE) ? 8 : -8;
    const int start_rank = (color == WHITE) ? 1 : 6;
    const int promo_rank = (color == WHITE) ? 6 : 1;
    const std::array<U64, 64> &pawn_attacks = (color == WHITE) ? PawnAttacksWhite : PawnAttacksBlack;

    U64 temp_pawns = pawns;
    while (temp_pawns)
    {
        const int from = cpu::pop_lsb(temp_pawns);
        const int rank = from >> 3;

        // 1. Poussées (Simple et Double)
        int to = from + dir;
        // Sécurité : on vérifie que 'to' est bien sur l'échiquier avant le test d'occupation
        if (to >= 0 && to < 64 && !(occ & (1ULL << to)))
        {
            if (rank == promo_rank)
            {
                list.push(Move{from, to, PAWN, Move::Flags::PROMOTION_MASK, NO_PIECE, QUEEN});
                list.push(Move{from, to, PAWN, Move::Flags::PROMOTION_MASK, NO_PIECE, KNIGHT});
                list.push(Move{from, to, PAWN, Move::Flags::PROMOTION_MASK, NO_PIECE, ROOK});
                list.push(Move{from, to, PAWN, Move::Flags::PROMOTION_MASK, NO_PIECE, BISHOP});
            }
            else
            {
                list.push(Move{from, to, PAWN, Move::Flags::NONE, NO_PIECE});

                // Double poussée
                int double_to = to + dir;
                if (rank == start_rank && !(occ & (1ULL << double_to)))
                {
                    list.push(Move{from, double_to, PAWN, Move::Flags::DOUBLE_PUSH, NO_PIECE});
                }
            }
        }

        // 2. Captures normales
        U64 attacks = pawn_attacks[from] & enemies;
        while (attacks)
        {
            int target = cpu::pop_lsb(attacks);
            const Piece target_p = board.get_p(target);
            if (rank == promo_rank)
            {
                list.push(Move{from, target, PAWN, Move::Flags::PROMOTION_MASK, target_p, QUEEN});
                list.push(Move{from, target, PAWN, Move::Flags::PROMOTION_MASK, target_p, KNIGHT});
                list.push(Move{from, target, PAWN, Move::Flags::PROMOTION_MASK, target_p, ROOK});
                list.push(Move{from, target, PAWN, Move::Flags::PROMOTION_MASK, target_p, BISHOP});
            }

            else
                list.push(Move{from, target, PAWN, Move::Flags::CAPTURE, target_p});
        }

        // 3. Capture En Passant
        if (ep_flag && (pawn_attacks[from] & (1ULL << ep_sq)))
        {
            list.push(Move{from, ep_sq, PAWN, Move::Flags::EN_PASSANT_CAP, PAWN});
        }
    }
}
template <Color Us>
void MoveGen::generate_pseudo_legal_moves(Board &board, MoveList &list)
{
    const U64 occupied = board.get_occupancy<NO_COLOR>();
    const U64 us_occ = board.get_occupancy<Us>();
    const U64 opponent_king_mask = ~board.get_piece_bitboard<!Us, KING>();

    // 1. Pions : Déjà dans une fonction dédiée (Excellent)
    generate_pawn_moves(board, Us, list);

    // 2. Cavaliers (Sauts fixes)
    U64 knights = board.get_piece_bitboard<Us, KNIGHT>();
    while (knights)
    {
        int sq = cpu::pop_lsb(knights);
        U64 targets = KnightAttacks[sq] & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, KNIGHT, targets, board);
    }

    // 3. Sliders : Fous et Tours (Utilisent l'occupancy)
    U64 bishops = board.get_piece_bitboard<Us, BISHOP>();
    while (bishops)
    {
        int sq = cpu::pop_lsb(bishops);
        U64 targets = generate_bishop_moves(sq, occupied) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, BISHOP, targets, board);
    }

    U64 rooks = board.get_piece_bitboard<Us, ROOK>();
    while (rooks)
    {
        int sq = cpu::pop_lsb(rooks);
        U64 targets = generate_rook_moves(sq, occupied) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, ROOK, targets, board);
    }

    // 4. Dames (Combinaison des deux)
    U64 queens = board.get_piece_bitboard<Us, QUEEN>();
    while (queens)
    {
        int sq = cpu::pop_lsb(queens);
        U64 targets = (generate_rook_moves(sq, occupied) | generate_bishop_moves(sq, occupied)) & ~us_occ & opponent_king_mask;
        push_moves_from_mask(list, sq, QUEEN, targets, board);
    }

    // 5. Roi
    const int sq = board.get_eval_state().king_sq[Us];
    U64 targets = KingAttacks[sq] & ~us_occ & opponent_king_mask;
    push_moves_from_mask(list, sq, KING, targets, board);

    generate_castle_moves<Us>(board, list);
}
template <Color Us, Piece P>
void generate_piece_captures(const Board &board, U64 opponent_occ, MoveList &list)
{
    // Récupération des pièces du type P pour la couleur donnée
    U64 pieces = board.get_piece_bitboard<Us, P>();

    // Optimisation : Si pas de pièces de ce type, on sort tout de suite
    if (!pieces)
        return;

    U64 valid_targets = opponent_occ;
    if constexpr (P == PAWN)
    {
        if (board.get_en_passant_sq() != constants::EnPassantSqNone)
        {
            valid_targets |= core::mask::sq_mask(board.get_en_passant_sq());
        }
    }

    while (pieces)
    {
        const int from_sq = cpu::pop_lsb(pieces);

        // Appel direct à la version template (très rapide)
        // On filtre directement avec les cibles valides (captures + EP)
        U64 attacks = MoveGen::get_pseudo_moves_mask<P>(board, from_sq, Us) & valid_targets; // TODO

        while (attacks)
        {
            const int to_sq = cpu::pop_lsb(attacks);

            Move m(from_sq, to_sq, P);
            MoveGen::init_move_flags(board, m);

            list.push(m);
        }
    }
}
template <Color Us>
void MoveGen::generate_castle_moves(Board &board, MoveList &list)
{
    const uint8_t castling_rights = board.get_castling_rights();
    const U64 occupancy = board.get_occupancy<NO_COLOR>();

    if constexpr (Us == WHITE)
    {
        // PETIT ROQUE BLANC (h1)
        // 1. Droits ? 2. Cases vides (f1, g1) ? 3. Sécurité (e1, f1, g1) ?
        if ((castling_rights & WHITE_KINGSIDE) && !(occupancy & 0x60ULL))
        {
            if (!is_mask_attacked<!Us>(board, 0x70ULL)) // e1, f1, g1
            {
                list.push(Move(4, 6, KING, Move::Flags::KING_CASTLE, NO_PIECE));
            }
        }
        // GRAND ROQUE BLANC (a1)
        // 1. Droits ? 2. Cases vides (b1, c1, d1) ? 3. Sécurité (c1, d1, e1) ?
        if ((castling_rights & WHITE_QUEENSIDE) && !(occupancy & 0x0EULL))
        {
            if (!is_mask_attacked<!Us>(board, 0x1CULL)) // c1, d1, e1
            {
                list.push(Move(4, 2, KING, Move::Flags::QUEEN_CASTLE, NO_PIECE));
            }
        }
    }
    else // BLACK
    {
        // PETIT ROQUE NOIR (h8)
        if ((castling_rights & BLACK_KINGSIDE) && !(occupancy & 0x6000000000000000ULL))
        {
            if (!is_mask_attacked<!Us>(board, 0x7000000000000000ULL)) // e8, f8, g8
            {
                list.push(Move(60, 62, KING, Move::Flags::KING_CASTLE, NO_PIECE));
            }
        }
        // GRAND ROQUE NOIR (a8)
        if ((castling_rights & BLACK_QUEENSIDE) && !(occupancy & 0x0E00000000000000ULL))
        {
            if (!is_mask_attacked<!Us>(board, 0x1C00000000000000ULL)) // c8, d8, e8
            {
                list.push(Move(60, 58, KING, Move::Flags::QUEEN_CASTLE, NO_PIECE));
            }
        }
    }
}

template <Color Us>
void MoveGen::generate_pseudo_legal_captures(const Board &board, MoveList &list)
{
    // On définit les cibles : uniquement les pièces adverses (pas le Roi pour éviter les pseudo-moves illégaux de capture de roi)

    // On retire le Roi adverse des cibles valides (car on ne capture jamais le roi)
    const U64 opponent_occ = board.get_occupancy<!Us>() & ~board.get_piece_bitboard<!Us, KING>();

    generate_piece_captures<Us, PAWN>(board, opponent_occ, list);
    generate_piece_captures<Us, KNIGHT>(board, opponent_occ, list);
    generate_piece_captures<Us, BISHOP>(board, opponent_occ, list);
    generate_piece_captures<Us, ROOK>(board, opponent_occ, list);
    generate_piece_captures<Us, QUEEN>(board, opponent_occ, list);
    generate_piece_captures<Us, KING>(board, opponent_occ, list);
}

/// @brief For GUI only (doesn't need high perfs)
/// @param board
/// @param from_sq
/// @return
U64 MoveGen::get_legal_moves_mask(Board &board, int from_sq)
{
    MoveList list;
    const Color player = board.get_side_to_move();
    PieceInfo info = board.get_piece_on_square(from_sq);
    if (info.second == NO_PIECE)
        return 0ULL;

    U64 target_mask_pseudo = get_pseudo_moves_mask(board, from_sq);
    U64 target_mask = 0ULL;
    while (target_mask_pseudo != 0)
    {
        int to_sq = cpu::pop_lsb(target_mask_pseudo);
        Move move(from_sq, to_sq, static_cast<Piece>(info.second));
        MoveGen::init_move_flags(board, move);
        if (board.is_move_legal(move))
            target_mask |= core::mask::sq_mask(to_sq);
    }

    if (info.second == KING)
    {
        if (player == WHITE)
        {
            generate_castle_moves<WHITE>(board, list);
        }
        else
        {
            generate_castle_moves<BLACK>(board, list);
        }

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

template <Color Attacker>
bool is_mask_attacked(const Board &board, U64 mask)
{

    const std::array<U64, constants::BoardSize> &PawnAttacks =
        (!Attacker == WHITE) ? MoveGen::PawnAttacksWhite : MoveGen::PawnAttacksBlack;

    const U64 occ = board.get_occupancy(NO_COLOR);

    // On boucle sur chaque case du masque (généralement 2 ou 3 cases pour le roque)
    while (mask)
    {
        int sq = cpu::pop_lsb(mask);

        // 1. Attaques de Pions (Reverse)
        // On regarde si un pion adverse sur sa case de départ pourrait attaquer 'sq'
        if (PawnAttacks[sq] & board.get_piece_bitboard<Attacker, PAWN>())
            return true;

        // 2. Attaques de Cavaliers
        if (MoveGen::KnightAttacks[sq] & board.get_piece_bitboard<Attacker, KNIGHT>())
            return true;

        // 3. Attaques de Fou / Dame (Diagonales)
        U64 bishop_attacks = MoveGen::generate_bishop_moves(sq, occ);
        if (bishop_attacks & (board.get_piece_bitboard<Attacker, BISHOP>() | board.get_piece_bitboard<Attacker, QUEEN>()))
            return true;

        // 4. Attaques de Tour / Dame (Colonnes/Rangées)
        U64 rook_attacks = MoveGen::generate_rook_moves(sq, occ);
        if (rook_attacks & (board.get_piece_bitboard<Attacker, ROOK>() | board.get_piece_bitboard<Attacker, QUEEN>()))
            return true;

        // 5. Attaques de Roi (Reverse)
        if (MoveGen::KingAttacks[sq] & board.get_piece_bitboard<Attacker, KING>())
            return true;
    }

    return false;
}

template <Color Us>
void MoveGen::generate_legal_moves(Board &board, MoveList &list)
{
    MoveGen::generate_pseudo_legal_moves<Us>(board, list);

    const bool is_king_atck = board.is_king_attacked<Us>();
    U64 king_bb = board.get_piece_bitboard<Us, KING>();
    const int king_sq = cpu::pop_lsb(king_bb);

    const U64 occ = board.get_occupancy<NO_COLOR>();

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
            ++write_idx;
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
            U64 target_mask = core::mask::sq_mask(target_sq);
            attacks |= target_mask;
            if (occupancy & target_mask) // bloqueur rencontré -> s'arrête
                break;
            r += d.first;
            f += d.second;
        }
    }
    return attacks;
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

#ifdef __BMI2__
    logs::debug << "BMI2 mode" << std::endl;

    MoveGen::load_attack_tables();

#else
    logs::debug << "Magic Numbers mode" << std::endl;

    MoveGen::load_magics();
    MoveGen::load_attacks();

#endif

    init_zobrist();

    logs::debug << "--- All U64 tables loaded successfully ! ---" << std::endl;
    return true;
}

static bool __bitboard_data_loaded = perform_initial_data_loading();

template void MoveGen::generate_pseudo_legal_captures<WHITE>(const Board &board, MoveList &list);
template void MoveGen::generate_pseudo_legal_captures<BLACK>(const Board &board, MoveList &list);
template void MoveGen::generate_legal_moves<WHITE>(Board &board, MoveList &list);
template void MoveGen::generate_legal_moves<BLACK>(Board &board, MoveList &list);