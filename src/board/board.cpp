#include "board.hpp"

#include <iostream>
#include <cassert>
#include <stdexcept>

void Board::clear()
{
    for (auto &b : pieces_occ)
    {
        b = 0;
    }
    occupied_white = 0;
    occupied_black = 0;
    occupied_all = 0;
    castling_rights = 0;
    en_passant_sq = 0;
    halfmove_clock = 0;
    side_to_move = WHITE;
    zobrist_key = 0;
    fullmove_number = 0;
    move_stack.clear();
}

std::pair<Color, Piece> Board::get_piece_on_square(int sq) const
{
    // Check if the square is occupied at all
    if (!is_set(occupied_all, sq))
    {
        return {NO_COLOR, NO_PIECE};
    }

    // Determine color first
    Color piece_color = is_set(occupied_white, sq) ? WHITE : BLACK;

    // Determine piece type
    for (int type = PAWN; type <= KING; ++type)
    {
        // Check if the piece's bitboard is set at the given square
        if (is_occupied(sq, type, piece_color))
        {
            return {piece_color, static_cast<Piece>(type)};
        }
    }

    throw std::logic_error("Incoherent result, square should be occupied");
}

bool Board::play(Move move)
{
    const int from_sq{move.get_from_sq()}, to_sq{move.get_to_sq()};
    const Piece from_piece{move.get_from_piece()};
    if (from_piece == NO_PIECE)
    {
        std::cerr << "No piece found on square" << from_sq << std::endl;
        return false;
    }
    const Color opponent_color = (side_to_move == WHITE) ? BLACK : WHITE;
    assert(is_occupied(from_sq, from_piece, side_to_move));
    assert(!is_occupied(to_sq, NO_PIECE, side_to_move));

    const U64 to_sq_mask{sq_mask(to_sq)};

    get_piece_bitboard(side_to_move, from_piece) ^= sq_mask(from_sq); // Delete old position
    get_piece_bitboard(side_to_move, from_piece) |= to_sq_mask;       // Fill new position

    // Checking if an opponent square has to be updated
    if (side_to_move == WHITE && !(occupied_black & to_sq_mask))
    {
        update_occupancy();
        switch_trait();
        return true;
    }

    if (side_to_move == BLACK && !(occupied_white & to_sq_mask))
    {
        update_occupancy();
        switch_trait();
        return true;
    }

    for (int i{PAWN}; i <= KING; ++i)
    {
        if (get_piece_bitboard(opponent_color, i) & to_sq_mask)
        {
            get_piece_bitboard(opponent_color, i) ^= to_sq_mask;
            update_occupancy();
            switch_trait();
            return true;
        }
    }

    throw std::logic_error("Occupancy mask and piece bitboards out of sync");
    return false;
}
