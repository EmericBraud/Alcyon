#pragma once

#include <utility>
#include <cstdint>

#include "common/constants.hpp"
#include "color.hpp"

enum Piece : std::uint8_t
{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    NO_PIECE,
};

using PieceInfo = std::pair<Color, Piece>;

constexpr inline int get_piece_index(Piece p, Color c)
{
    return p + constants::PieceTypeCount * c;
}
