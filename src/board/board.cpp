#include "board.hpp"

#include <iostream>

void Board::clear()
{
    for (auto &b : piece_bitboards)
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
    if (!(occupied_all & (1ULL << sq)))
    {
        return {NO_COLOR, NO_PIECE};
    }

    // Determine color first
    Color piece_color = (occupied_white & (1ULL << sq)) ? WHITE : BLACK;

    // Determine piece type
    for (int type = PAWN; type <= KING; ++type)
    {

        // This index mapping must be consistent with your array:
        // WHITE: indices 0 to 5 (PAWN=1 to KING=6)
        // BLACK: indices 6 to 11 (PAWN=1 to KING=6)

        size_t index = (piece_color * N_PIECES_TYPE_HALF) + (type - 1);

        // Check if the piece's bitboard is set at the given square
        if (piece_bitboards[index] & (1ULL << sq))
        {
            return {piece_color, (Piece)type};
        }
    }

    // Should not be reached if occupied_all check was correct
    throw std::logic_error("Incoherent result, square should be occupied");
}
