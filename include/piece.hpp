#pragma once

#include "bitboard.hpp"

enum Piece : uint8_t
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    NO_PIECE,
};

enum Color : uint8_t
{
    WHITE,
    BLACK,
    NO_COLOR
};

using PieceInfo = std::pair<Color, Piece>;

inline int get_piece_index(const Piece p, const Color c)
{
    return p + N_PIECES_TYPE_HALF * c;
}
