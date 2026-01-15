#include "core/board.hpp"
#include "../move/move_generator.cpp"

// clang-format off
static constexpr uint8_t CastlingMask[64] = {
    static_cast<uint8_t>(~WHITE_QUEENSIDE),                    15, 15, 15,
    static_cast<uint8_t>(~(WHITE_QUEENSIDE | WHITE_KINGSIDE)), 15, 15, static_cast<uint8_t>(~WHITE_KINGSIDE),
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 15, 15,
    static_cast<uint8_t>(~BLACK_QUEENSIDE),                    15, 15, 15,
    static_cast<uint8_t>(~(BLACK_QUEENSIDE | BLACK_KINGSIDE)), 15, 15, static_cast<uint8_t>(~BLACK_KINGSIDE),
};

namespace MoveGen{
    bool is_king_attacked(const Board &board, Color us);
}
// clang-format on

void Board::clear()
{
    for (auto &b : pieces_occ)
    {
        b = 0;
    }
    occupancies[WHITE] = 0;
    occupancies[BLACK] = 0;
    occupancies[NO_COLOR] = 0;
    state.castling_rights = 0;
    state.en_passant_sq = core::constants::EnPassantSqNone;
    state.halfmove_clock = 0;
    state.side_to_move = WHITE;
    state.last_irreversible_index = 0;
    zobrist_key = 0;
    if (history_tagged)
        get_history()->clear();
    std::memset(mailbox, EMPTY_SQ, core::constants::BoardSize);
}
template <Color Us>
bool Board::play(const Move move)
{
    // 1. Sauvegarde avant modification
    get_history()->push_back({zobrist_key, state.halfmove_clock, state.last_irreversible_index, move, state.en_passant_sq, state.castling_rights});

    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();
    const Color them = (Color)!Us;

    bool is_irreversible = (from_piece == PAWN) || (to_piece != NO_PIECE);
    eval_state.increment(move, Us);

    if (is_irreversible)
    {
        state.last_irreversible_index = (int)get_history()->size() - 1;
        state.halfmove_clock = 0;
    }
    else
    {
        state.halfmove_clock++;
    }

    // Mise à jour Zobrist (Phase 1 : Retrait état actuel)
    zobrist_key ^= zobrist_castling[state.castling_rights];
    zobrist_key ^= zobrist_en_passant[state.en_passant_sq == core::constants::EnPassantSqNone ? 8 : state.en_passant_sq % 8];

    // 2. Mouvement de la pièce
    const U64 from_mask = 1ULL << from_sq;
    const U64 to_mask = 1ULL << to_sq;

    zobrist_key ^= zobrist_table[Us * core::constants::PieceTypeCount + from_piece][from_sq];
    get_piece_bitboard(Us, from_piece) ^= from_mask;
    mailbox[from_sq] = EMPTY_SQ;

    // 3. Gestion de la capture classique
    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP)
    {
        zobrist_key ^= zobrist_table[them * core::constants::PieceTypeCount + to_piece][to_sq];
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask; // Mise à jour directe de l'adversaire
    }

    state.en_passant_sq = core::constants::EnPassantSqNone;
    Piece final_piece = from_piece;

    // 4. Cas spéciaux
    if (flags) [[unlikely]]
    {
        switch (flags)
        {
        case Move::Flags::PROMOTION_MASK:
            final_piece = move.get_promo_piece();
            break;
        case Move::Flags::DOUBLE_PUSH:
            state.en_passant_sq = (from_sq + to_sq) >> 1;
            break;
        case Move::Flags::EN_PASSANT_CAP:
        {
            const int cap_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
            zobrist_key ^= zobrist_table[them * core::constants::PieceTypeCount + PAWN][cap_sq];
            get_piece_bitboard<them, PAWN>() ^= (1ULL << cap_sq);
            occupancies[them] ^= (1ULL << cap_sq);
            mailbox[cap_sq] = EMPTY_SQ;
            break;
        }
        case Move::Flags::KING_CASTLE:
        case Move::Flags::QUEEN_CASTLE:
        {
            bool is_ks = (flags == Move::Flags::KING_CASTLE);
            int r_f = is_ks ? (Us == WHITE ? 7 : 63) : (Us == WHITE ? 0 : 56);
            int r_t = is_ks ? (Us == WHITE ? 5 : 61) : (Us == WHITE ? 3 : 59);
            U64 r_mask = (1ULL << r_f) | (1ULL << r_t);
            zobrist_key ^= zobrist_table[Us * core::constants::PieceTypeCount + ROOK][r_f] ^ zobrist_table[Us * core::constants::PieceTypeCount + ROOK][r_t];
            get_piece_bitboard<Us, ROOK>() ^= r_mask;
            occupancies[Us] ^= r_mask;
            mailbox[r_f] = EMPTY_SQ;
            mailbox[r_t] = (Us << COLOR_SHIFT) | ROOK;
            break;
        }
        }
    }

    // 5. Finalisation
    zobrist_key ^= zobrist_table[Us * core::constants::PieceTypeCount + final_piece][to_sq];
    get_piece_bitboard(Us, final_piece) |= to_mask;
    mailbox[to_sq] = (Us << COLOR_SHIFT) | final_piece;

    // Mise à jour des occupations globales (Vital pour is_attacked et PinTest)
    occupancies[Us] ^= (from_mask | to_mask);
    occupancies[NO_COLOR] = occupancies[WHITE] | occupancies[BLACK];

    state.castling_rights &= CastlingMask[from_sq] & CastlingMask[to_sq];
    zobrist_key ^= zobrist_castling[state.castling_rights];
    zobrist_key ^= zobrist_en_passant[state.en_passant_sq == core::constants::EnPassantSqNone ? 8 : state.en_passant_sq % 8];

    if (from_piece == KING)
        eval_state.king_sq[Us] = to_sq;
    switch_trait();
    return true;
}
template <Color Us>
void Board::unplay(const Move move)
{
    const uint32_t flags = move.get_flags();
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();
    const Piece moved_p = (flags == Move::Flags::PROMOTION_MASK) ? move.get_promo_piece() : from_piece;

    switch_trait();
    const Color them = (Color)!Us;

    // 1. Inversion Bitboards
    U64 move_mask = (1ULL << from_sq) | (1ULL << to_sq);
    get_piece_bitboard(Us, moved_p) ^= move_mask;
    occupancies[Us] ^= move_mask;

    // 2. Inversion Mailbox
    mailbox[from_sq] = (Us << COLOR_SHIFT) | from_piece;
    if (flags == Move::Flags::EN_PASSANT_CAP)
    {
        mailbox[to_sq] = EMPTY_SQ;
        const int cap_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
        get_piece_bitboard<them, PAWN>() |= (1ULL << cap_sq);
        occupancies[them] |= (1ULL << cap_sq);
        mailbox[cap_sq] = (them << COLOR_SHIFT) | PAWN;
    }
    else
    {
        mailbox[to_sq] = (to_piece == NO_PIECE) ? EMPTY_SQ : (them << COLOR_SHIFT) | to_piece;
        if (to_piece != NO_PIECE)
        {
            get_piece_bitboard(them, to_piece) |= (1ULL << to_sq);
            occupancies[them] |= (1ULL << to_sq);
        }
    }

    // 3. Cas spéciaux (Roque et Promotion)
    if (flags == Move::Flags::KING_CASTLE || flags == Move::Flags::QUEEN_CASTLE)
    {
        bool is_ks = (flags == Move::Flags::KING_CASTLE);
        int r_f = is_ks ? (Us == WHITE ? 7 : 63) : (Us == WHITE ? 0 : 56);
        int r_t = is_ks ? (Us == WHITE ? 5 : 61) : (Us == WHITE ? 3 : 59);
        U64 r_mask = (1ULL << r_f) | (1ULL << r_t);
        get_piece_bitboard<Us, ROOK>() ^= r_mask;
        occupancies[Us] ^= r_mask;
        mailbox[r_f] = (Us << COLOR_SHIFT) | ROOK;
        mailbox[r_t] = EMPTY_SQ;
    }
    else if (flags == Move::Flags::PROMOTION_MASK)
    {
        get_piece_bitboard(Us, move.get_promo_piece()) ^= (1ULL << from_sq);
        get_piece_bitboard<Us, PAWN>() |= (1ULL << from_sq);
    }

    // 4. Sync et Restauration
    occupancies[NO_COLOR] = occupancies[WHITE] | occupancies[BLACK];
    if (from_piece == KING)
        eval_state.king_sq[Us] = from_sq;

    const UndoInfo &info = get_history()->back();
    zobrist_key = info.zobrist_key;
    state.halfmove_clock = info.halfmove_clock;
    state.last_irreversible_index = info.last_irreversible_index;
    state.castling_rights = info.castling_rights;
    state.en_passant_sq = info.en_passant_sq;

    eval_state.decrement(move, Us);

    get_history()->pop_back();
}
bool Board::is_repetition() const
{
    const int n = (int)get_history()->size();
    // On recule de 2 en 2 car la répétition doit être au même trait
    // On s'arrête au dernier coup irréversible
    for (int i = n - 2; i >= state.last_irreversible_index; i -= 2)
    {
        if ((*get_history())[i].zobrist_key == zobrist_key)
        {
            return true;
        }
    }
    return false;
}

template bool Board::play<WHITE>(const Move move);
template bool Board::play<BLACK>(const Move move);

template void Board::unplay<WHITE>(const Move move);
template void Board::unplay<BLACK>(const Move move);

bool Board::is_move_pseudo_legal(const Move &move) const
{
    // 1. Vérifications de base (Sanity Checks)
    const int from = move.get_from_sq();
    const int to = move.get_to_sq();
    const Piece p = move.get_from_piece();
    const Color us = state.side_to_move;

    // La pièce de départ doit exister et appartenir au joueur dont c'est le tour
    // Note : get_piece_at et get_color_at sont supposées exister dans votre Board
    // Si vous stockez les bitboards, vous pouvez vérifier :
    // if (!(get_piece_bitboard(us, p) & (1ULL << from))) return false;
    PieceInfo info = get_piece_on_square(from); // Votre fonction existante
    if (info.second != p || info.first != us)
        return false;

    // La case d'arrivée ne doit pas contenir une de nos propres pièces
    // (Capture amicale impossible)
    if (occupancies[us] & (1ULL << to))
        return false;

    const U64 occ = occupancies[NO_COLOR];
    const U64 to_mask = (1ULL << to);

    // 2. Logique par type de pièce
    switch (p)
    {
    case KNIGHT:
        // Vérifie si 'to' est dans la table d'attaque du cavalier depuis 'from'
        return MoveGen::KnightAttacks[from] & to_mask;

    case KING:
        // Cas spécial : Roque
        if (move.get_flags() == Move::Flags::KING_CASTLE || move.get_flags() == Move::Flags::QUEEN_CASTLE)
        {
            // Vérification des droits de roque et des cases vides
            // (La sécurité du roi est gérée par is_move_legal, donc on vérifie juste la mécanique)
            if (us == WHITE)
            {
                if (move.get_flags() == Move::Flags::KING_CASTLE)
                    return (state.castling_rights & WHITE_KINGSIDE) && !(occ & 0x60ULL) && (to == 6);
                else
                    return (state.castling_rights & WHITE_QUEENSIDE) && !(occ & 0x0EULL) && (to == 2);
            }
            else
            {
                if (move.get_flags() == Move::Flags::KING_CASTLE)
                    return (state.castling_rights & BLACK_KINGSIDE) && !(occ & 0x6000000000000000ULL) && (to == 62);
                else
                    return (state.castling_rights & BLACK_QUEENSIDE) && !(occ & 0x0E00000000000000ULL) && (to == 58);
            }
        }
        // Mouvement normal du roi
        return MoveGen::KingAttacks[from] & to_mask;

    case BISHOP:
        // Utilisation des Magics pour vérifier si le chemin est libre
        return MoveGen::generate_bishop_moves(from, occ) & to_mask;

    case ROOK:
        return MoveGen::generate_rook_moves(from, occ) & to_mask;

    case QUEEN:
        return (MoveGen::generate_bishop_moves(from, occ) | MoveGen::generate_rook_moves(from, occ)) & to_mask;

    case PAWN:
    {
        const int dir = (us == WHITE) ? 8 : -8;
        const int start_rank = (us == WHITE) ? 1 : 6;

        // a. Poussée simple
        if (to == from + dir)
        {
            // La case doit être vide
            return !(occ & to_mask) && (move.get_flags() == Move::Flags::NONE || move.get_flags() == Move::Flags::PROMOTION_MASK);
        }

        // b. Double poussée
        if (move.get_flags() == Move::Flags::DOUBLE_PUSH)
        {
            if ((from / 8) != start_rank)
                return false;
            if (to != from + 2 * dir)
                return false;
            // Case intermédiaire et finale doivent être vides
            return !(occ & to_mask) && !(occ & (1ULL << (from + dir)));
        }

        // c. Captures (y compris En Passant)
        // Vérifier que 'to' est une diagonale valide
        const U64 pawn_attacks = (us == WHITE) ? MoveGen::PawnAttacksWhite[from] : MoveGen::PawnAttacksBlack[from];
        if (!(pawn_attacks & to_mask))
            return false;

        // Si capture normale : doit y avoir un ennemi
        if (move.get_flags() == Move::Flags::CAPTURE || move.get_flags() == Move::Flags::PROMOTION_MASK)
        {
            return (occupancies[!us] & to_mask);
        }

        // Si En Passant
        if (move.get_flags() == Move::Flags::EN_PASSANT_CAP)
        {
            return (state.en_passant_sq != core::constants::EnPassantSqNone) && (to == state.en_passant_sq);
        }

        return false;
    }

    default:
        return false;
    }
}