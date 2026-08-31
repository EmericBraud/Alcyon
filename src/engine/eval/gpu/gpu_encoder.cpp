#include "gpu_encoder.hpp"

#include <cstring>

#include "core/move/generator/move_generator.hpp"

namespace gpu_eval {

namespace {

void fill_plane_zero(float *plane) { std::memset(plane, 0, sizeof(float) * kPlaneSize); }

void fill_plane_broadcast(float *plane, float value) {
    for (int sq = 0; sq < kPlaneSize; ++sq) {
        plane[sq] = value;
    }
}

// Union of pseudo-legal attacks (including squares occupied by own pieces,
// matching plane_batch.cpp's python-chess-derived semantics exactly) for
// every piece of `piece` and `color` on this position.
U64 attacks_for(const GpuPosition &pos, Color color, Piece piece, U64 occ) {
    U64 bb = pos.piece_bb(color, piece);
    U64 attacked = 0;
    while (bb) {
        const int from = std::countr_zero(bb);
        bb &= bb - 1;
        switch (piece) {
        case PAWN:
            attacked |= (color == WHITE ? MoveGen::PawnAttacksWhite[from] : MoveGen::PawnAttacksBlack[from]);
            break;
        case KNIGHT:
            attacked |= MoveGen::KnightAttacks[from];
            break;
        case BISHOP:
            attacked |= MoveGen::generate_bishop_moves(from, occ);
            break;
        case ROOK:
            attacked |= MoveGen::generate_rook_moves(from, occ);
            break;
        case QUEEN:
            attacked |= MoveGen::generate_rook_moves(from, occ) | MoveGen::generate_bishop_moves(from, occ);
            break;
        case KING:
            attacked |= MoveGen::KingAttacks[from];
            break;
        default:
            break;
        }
    }
    return attacked;
}

} // namespace

void encode_planes_v3(const GpuPosition &pos, float planes[kNumPlanesV3][kPlaneSize]) {
    const Color us = pos.side_to_move;
    const Color them = !us;
    const bool orient_flip = (us == BLACK);
    const U64 occ = pos.all_occupancy();

    // 0-5 our pieces, 6-11 their pieces.
    for (int p = 0; p < constants::PieceTypeCount; ++p) {
        fill_plane_zero(planes[p]);
        fill_plane_zero(planes[p + 6]);

        U64 bb_us = pos.piece_bb(us, static_cast<Piece>(p));
        while (bb_us) {
            const int sq = std::countr_zero(bb_us);
            bb_us &= bb_us - 1;
            planes[p][orient_flip ? flip_rank(sq) : sq] = 1.0f;
        }

        U64 bb_them = pos.piece_bb(them, static_cast<Piece>(p));
        while (bb_them) {
            const int sq = std::countr_zero(bb_them);
            bb_them &= bb_them - 1;
            planes[p + 6][orient_flip ? flip_rank(sq) : sq] = 1.0f;
        }
    }

    // 12: side to move is white.
    fill_plane_broadcast(planes[12], us == WHITE ? 1.0f : 0.0f);

    // 13-16: castling rights (us kingside, us queenside, them kingside, them queenside).
    const std::uint8_t our_kingside = (us == WHITE) ? WHITE_KINGSIDE : BLACK_KINGSIDE;
    const std::uint8_t our_queenside = (us == WHITE) ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
    const std::uint8_t their_kingside = (them == WHITE) ? WHITE_KINGSIDE : BLACK_KINGSIDE;
    const std::uint8_t their_queenside = (them == WHITE) ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
    fill_plane_broadcast(planes[13], (pos.castling_rights & our_kingside) ? 1.0f : 0.0f);
    fill_plane_broadcast(planes[14], (pos.castling_rights & our_queenside) ? 1.0f : 0.0f);
    fill_plane_broadcast(planes[15], (pos.castling_rights & their_kingside) ? 1.0f : 0.0f);
    fill_plane_broadcast(planes[16], (pos.castling_rights & their_queenside) ? 1.0f : 0.0f);

    // 17: en passant target square.
    fill_plane_zero(planes[17]);
    if (pos.en_passant_sq != constants::EnPassantSqNone) {
        const int sq = pos.en_passant_sq;
        planes[17][orient_flip ? flip_rank(sq) : sq] = 1.0f;
    }

    // 18: halfmove clock (rule50), normalized.
    fill_plane_broadcast(planes[18], static_cast<float>(pos.halfmove_clock) / 100.0f);

    // 19-24 our attacks, 25-30 their attacks, per piece type.
    for (int p = 0; p < constants::PieceTypeCount; ++p) {
        const U64 us_attacks = attacks_for(pos, us, static_cast<Piece>(p), occ);
        const U64 them_attacks = attacks_for(pos, them, static_cast<Piece>(p), occ);

        fill_plane_zero(planes[19 + p]);
        fill_plane_zero(planes[25 + p]);

        U64 bb = us_attacks;
        while (bb) {
            const int sq = std::countr_zero(bb);
            bb &= bb - 1;
            planes[19 + p][orient_flip ? flip_rank(sq) : sq] = 1.0f;
        }
        bb = them_attacks;
        while (bb) {
            const int sq = std::countr_zero(bb);
            bb &= bb - 1;
            planes[25 + p][orient_flip ? flip_rank(sq) : sq] = 1.0f;
        }
    }
}

} // namespace gpu_eval
