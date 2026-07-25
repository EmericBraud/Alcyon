#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"

#include "common/mask.hpp"

template <Color Attacker>
bool Board::is_attacked(int sq) const
{
    const U64 occupied = occupancies[NO_COLOR];

    // 1. Pions et Cavaliers (Les plus rapides, on commence par eux)
    if ((Attacker == WHITE ? MoveGen::PawnAttacksBlack[sq] : MoveGen::PawnAttacksWhite[sq]) & get_piece_bitboard<Attacker, PAWN>())
        return true;

    if (MoveGen::KnightAttacks[sq] & get_piece_bitboard<Attacker, KNIGHT>())
        return true;

    // 2. Roi (Très rapide aussi)
    if (MoveGen::KingAttacks[sq] & get_piece_bitboard<Attacker, KING>())
        return true;

    // 3. Sliders : On combine les bitboards de l'attaquant pour limiter les tests
    const U64 queens = get_piece_bitboard<Attacker, QUEEN>();

    // Diagonales (Fous + Dames)
    if (MoveGen::generate_bishop_moves(sq, occupied) & (get_piece_bitboard<Attacker, BISHOP>() | queens))
        return true;

    // Lignes/Colonnes (Tours + Dames)
    if (MoveGen::generate_rook_moves(sq, occupied) & (get_piece_bitboard<Attacker, ROOK>() | queens))
        return true;

    return false;
}
/*
 * Plays and unplays the move without considering zobrist key / EvalState, just in order to check if the move is legal
 * (assuming that the move is already pseudo-legal)
 */
template <Color Us>
bool Board::is_move_legal(const Move move)
{
    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();
    constexpr Color them = (Color)!Us;

    const U64 from_mask = 1ULL << from_sq;
    const U64 to_mask = 1ULL << to_sq;
    const U64 move_mask = from_mask | to_mask;

    if (flags == Move::Flags::KING_CASTLE) [[unlikely]]
    {
        if (is_king_attacked<Us>())
            return false;
        if (is_attacked<!Us>(king_sq[Us] + 1))
            return false;
    }
    else if (flags == Move::Flags::QUEEN_CASTLE) [[unlikely]]
    {
        if (is_king_attacked<Us>())
            return false;
        if (is_attacked<!Us>(king_sq[Us] - 1))
            return false;
    }

    get_piece_bitboard(Us, from_piece) ^= move_mask;
    occupancies[Us] ^= move_mask;
    occupancies[NO_COLOR] ^= move_mask;

    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask;
        occupancies[NO_COLOR] ^= to_mask;
    }

    U64 ep_pawn_mask = 0;
    if (flags == Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        int ep_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
        ep_pawn_mask = 1ULL << ep_sq;
        get_piece_bitboard(them, PAWN) ^= ep_pawn_mask;
        occupancies[them] ^= ep_pawn_mask;
        occupancies[NO_COLOR] ^= ep_pawn_mask;
    }
    int32_t is_king_mask = -(from_piece == KING);
    king_sq[Us] ^= (to_sq ^ king_sq[Us]) & is_king_mask;

    bool legal = !is_king_attacked<Us>();
    // If king is moved
    int32_t mask = -(from_piece == KING);
    king_sq[Us] ^= (from_sq ^ king_sq[Us]) & mask;
    // 4. Restauration (Ordre inverse exact)
    if (flags == Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, PAWN) ^= ep_pawn_mask;
        occupancies[them] ^= ep_pawn_mask;
        occupancies[NO_COLOR] ^= ep_pawn_mask;
    }

    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask;
        occupancies[NO_COLOR] ^= to_mask;
    }

    get_piece_bitboard(Us, from_piece) ^= move_mask;
    occupancies[Us] ^= move_mask;
    occupancies[NO_COLOR] ^= move_mask;

    return legal;
}

// Same shape as is_move_legal() above (toggle raw bitboards, test, toggle
// back -- no zobrist/eval_state/NNUE touched), but tests whether the move
// attacks Them's king instead of whether Us's own king ends up safe. Used
// by futility pruning's "does this quiet move give check" check
// (negamax.cpp) to avoid a real play()/unplay() round trip -- which, in an
// NNUE build, would otherwise pay for a full incremental accumulator
// update and its reversal just to answer this one question.
template <Color Us>
bool Board::gives_check(const Move move)
{
    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();
    // On promotion, get_to_piece() reports the *captured* piece (or
    // NO_PIECE), not what ends up on to_sq -- the pawn is replaced by
    // get_promo_piece() instead, unlike every other move type where
    // to_sq's post-move occupant is from_piece. Getting this wrong matters
    // here (unlike in is_move_legal() above, which this function otherwise
    // mirrors): a promoted queen/rook/bishop routinely gives check where a
    // pawn wouldn't, so using from_piece would misclassify exactly the
    // capture-adjacent moves futility pruning most needs right.
    const Piece placed_piece = (flags == Move::Flags::PROMOTION_MASK) ? move.get_promo_piece() : from_piece;
    constexpr Color them = (Color)!Us;

    const U64 from_mask = 1ULL << from_sq;
    const U64 to_mask = 1ULL << to_sq;

    get_piece_bitboard(Us, from_piece) ^= from_mask;
    get_piece_bitboard(Us, placed_piece) ^= to_mask;
    occupancies[Us] ^= from_mask | to_mask;
    occupancies[NO_COLOR] ^= from_mask | to_mask;

    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask;
        occupancies[NO_COLOR] ^= to_mask;
    }

    U64 ep_pawn_mask = 0;
    if (flags == Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        int ep_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
        ep_pawn_mask = 1ULL << ep_sq;
        get_piece_bitboard(them, PAWN) ^= ep_pawn_mask;
        occupancies[them] ^= ep_pawn_mask;
        occupancies[NO_COLOR] ^= ep_pawn_mask;
    }
    int32_t is_king_mask = -(from_piece == KING);
    king_sq[Us] ^= (to_sq ^ king_sq[Us]) & is_king_mask;

    bool check = is_king_attacked<them>();

    // If king is moved
    int32_t mask = -(from_piece == KING);
    king_sq[Us] ^= (from_sq ^ king_sq[Us]) & mask;
    // Restauration (Ordre inverse exact)
    if (flags == Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, PAWN) ^= ep_pawn_mask;
        occupancies[them] ^= ep_pawn_mask;
        occupancies[NO_COLOR] ^= ep_pawn_mask;
    }

    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask;
        occupancies[NO_COLOR] ^= to_mask;
    }

    get_piece_bitboard(Us, placed_piece) ^= to_mask;
    get_piece_bitboard(Us, from_piece) ^= from_mask;
    occupancies[Us] ^= from_mask | to_mask;
    occupancies[NO_COLOR] ^= from_mask | to_mask;

    return check;
}

template bool Board::is_attacked<WHITE>(int sq) const;
template bool Board::is_attacked<BLACK>(int sq) const;

template bool Board::is_move_legal<WHITE>(const Move move);
template bool Board::is_move_legal<BLACK>(const Move move);

template bool Board::gives_check<WHITE>(const Move move);
template bool Board::gives_check<BLACK>(const Move move);