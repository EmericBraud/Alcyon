#pragma once

#include <array>
#include <bit>

#include "core/piece/piece.hpp"
#include "gpu_position.hpp"

// Native mirror of training/cnn/data_loader/plane_batch.cpp's plane layout
// -- keep the two in sync. v3 (the checkpoint this module targets, see
// training/cnn/legacy_model.py's construction recipe) predates the
// king-distance planes added in v4, so this only builds planes 0-30 (31
// planes total, NUM_PLANES for v3), NOT the full 33-plane layout used by
// v4/v5/v6.
//
//   0-5   : our pieces   (pawn, knight, bishop, rook, queen, king)
//   6-11  : their pieces (same order)
//   12    : side to move is white (constant plane)
//   13-16 : castling rights (us kingside, us queenside, them kingside, them queenside)
//   17    : en passant target square (one-hot)
//   18    : halfmove clock (rule50), normalized to [0, 1]
//   19-24 : squares attacked by our pawn/knight/bishop/rook/queen/king
//   25-30 : same, for their pieces
namespace gpu_eval {

inline constexpr int kNumPlanesV3 = 31;
inline constexpr int kBoardSize = 8;
inline constexpr int kPlaneSize = kBoardSize * kBoardSize;

// v3's phase-bucket scheme (4 buckets, see model_loader.py's fix for v1/v3
// checkpoints -- NOT v4's 8-bucket scheme). Boundaries are on NON-KING
// piece count, both sides.
inline constexpr int kNumPhaseBucketsV3 = 4;
inline constexpr int kPhaseBucketBoundariesV3[3] = {24, 16, 8};

inline int phase_bucket_v3(int non_king_piece_count) {
    int bucket = 0;
    for (int boundary : kPhaseBucketBoundariesV3) {
        if (non_king_piece_count < boundary) {
            ++bucket;
        }
    }
    return bucket;
}

constexpr int flip_rank(int square) { return square ^ 56; }

// planes: row-major (plane, square), square = rank*8+file, a1=0 -- same
// convention the engine itself and the training pipeline both already use
// (see gpu_position.hpp / plane_batch.cpp), so no reindexing needed beyond
// the side-to-move rank flip below.
void encode_planes_v3(const GpuPosition &pos, float planes[kNumPlanesV3][kPlaneSize]);

// Non-king piece count, both sides -- matches training's plane_batch.h
// piece_count field (engine code elsewhere popcounts occupancy<NO_COLOR>()
// which INCLUDES kings; this one deliberately excludes them).
inline int non_king_piece_count(const GpuPosition &pos) {
    U64 occ = pos.all_occupancy() & ~(pos.piece_bb(WHITE, KING) | pos.piece_bb(BLACK, KING));
    return std::popcount(occ);
}

} // namespace gpu_eval
