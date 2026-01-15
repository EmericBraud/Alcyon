#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"

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
bool Board::is_move_legal(const Move move)
{
    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();
    const Color us = state.side_to_move;
    const Color them = (Color)!us;

    // 1. Préparation des masques
    const U64 from_mask = 1ULL << from_sq;
    const U64 to_mask = 1ULL << to_sq;
    const U64 move_mask = from_mask | to_mask;

    // 2. Simulation du mouvement (Bitboards de base uniquement)
    // On déplace la pièce
    get_piece_bitboard(us, from_piece) ^= move_mask;
    occupancies[us] ^= move_mask;
    occupancies[NO_COLOR] ^= move_mask;

    // Gestion de la capture classique
    if (to_piece != NO_PIECE && flags != Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        get_piece_bitboard(them, to_piece) ^= to_mask;
        occupancies[them] ^= to_mask;
        // CORRECTION : Rétablir l'occupation sur to_sq car le fou s'y trouve maintenant !
        occupancies[NO_COLOR] ^= to_mask;
    }

    // Cas spéciaux (EP et Roque)
    U64 ep_pawn_mask = 0;
    if (flags == Move::Flags::EN_PASSANT_CAP) [[unlikely]]
    {
        int ep_sq = (us == WHITE) ? to_sq - 8 : to_sq + 8;
        ep_pawn_mask = 1ULL << ep_sq;
        get_piece_bitboard(them, PAWN) ^= ep_pawn_mask;
        occupancies[them] ^= ep_pawn_mask;
        occupancies[NO_COLOR] ^= ep_pawn_mask;
    }
    // If king is moving, we have to update the king square (using arithmetic branchless expression)
    int32_t is_king_mask = -(from_piece == KING);
    king_sq[us] ^= (to_sq ^ king_sq[us]) & is_king_mask;
    // Note : Le roque se vérifie généralement AVANT d'appeler is_move_legal
    // car il a ses propres règles (ne pas être en échec, etc.)

    // 3. Vérification de la sécurité du Roi
    // On doit mettre à jour state.side_to_move pour que is_king_attacked regarde le bon camp
    // Mais il est plus rapide de passer la couleur en paramètre si ta fonction le permet
    bool legal = !is_king_attacked(us);
    // If king is moved
    int32_t mask = -(from_piece == KING);
    king_sq[us] ^= (from_sq ^ king_sq[us]) & mask;
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

    get_piece_bitboard(us, from_piece) ^= move_mask;
    occupancies[us] ^= move_mask;
    occupancies[NO_COLOR] ^= move_mask;

    return legal;
}

template bool Board::is_attacked<WHITE>(int sq) const;
template bool Board::is_attacked<BLACK>(int sq) const;