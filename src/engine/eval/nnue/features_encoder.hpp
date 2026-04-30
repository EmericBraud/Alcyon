#pragma once

#pragma once

#include <cassert>

#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"

namespace feature_encoder
{
    template <Color color>
        requires(color != NO_COLOR)
    int get_feature_index(int king_sq, Color piece_color, Piece piece_type, int piece_sq)
    {
        assert(piece_type != KING);
        const int ksq = color == WHITE ? king_sq : (king_sq ^ 56);
        const int psq = color == WHITE ? piece_sq : (piece_sq ^ 56);

        const int base = (piece_color == color) ? 0 : 5;
        const int piece_index = base + static_cast<int>(piece_type);

        return ((ksq * 10 + piece_index) << 6) + psq;
    }
}