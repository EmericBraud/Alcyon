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
        assert(piece_type != NO_PIECE);

        constexpr int piece_square_span = 64;
        constexpr int piece_types_per_bucket = 11;

        const int flip = (color == WHITE) ? 0 : 56;
        const int oriented_king_sq = king_sq ^ flip;
        const int oriented_piece_sq = piece_sq ^ flip;

        const int king_file = oriented_king_sq % 8;
        const int mirrored_file = (king_file < 4) ? king_file : (7 - king_file);
        const int king_rank = oriented_king_sq / 8;
        const int king_bucket = (7 - king_rank) * 4 + mirrored_file;

        const int p_idx = (piece_type == KING)
                              ? 10
                              : static_cast<int>(piece_type) * 2 + (piece_color != color ? 1 : 0);

        int final_piece_sq = oriented_piece_sq;
        if (king_file > 3)
            final_piece_sq ^= 7;

        return (king_bucket * piece_types_per_bucket * piece_square_span) + (p_idx * piece_square_span) + final_piece_sq;
    }
}