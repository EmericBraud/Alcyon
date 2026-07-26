#pragma once

// Port of nnue-pytorch's HalfKAv2_hm feature set (model/modules/features/halfka_v2_hm.py
// and the C++ mirror in data_loader/cpp/training_data_loader.cpp), at commit
// 4289208fe20cc6ec8753e5ee14c2f210de783ff0.
//
// This computes the *real* (post-coalesce, post 12->11 piece-type remap) feature
// index directly -- i.e. the index space actually stored in a serialized .nnue
// file (NUM_REAL_FEATURES = 704 * 32 = 22,528), without going through the
// 24,576-wide virtual/king-independent factorizer space (that space only exists
// during training and is coalesced away before export).
//
// Derivation (see halfka_v2_hm.py):
//   - _orient(is_white_pov, sq, ksq): sq ^= (7 if ksq.file() < 4 else 0) ^ (56 if
//     not is_white_pov else 0). I.e. mirror the file so the king always ends up
//     on the e-h side, and flip the whole board vertically for Black's POV.
//   - King bucket: KingBuckets[oriented king square] (32 buckets, only valid --
//     i.e. non-negative -- for files e-h, which orientation guarantees).
//   - p_idx (virtual, 12-wide): (piece_type - 1) * 2 + (piece_color != pov), with
//     Pawn=1..King=6 in python numbering. In this codebase's 0-indexed Piece enum
//     (PAWN=0..KING=5) that's simply piece_type * 2 + (piece_color != pov).
//   - Virtual index: oriented_sq + p_idx * 64 + bucket * 768.
//   - Export (real, 11-wide) remap: the first 10 planes (p_idx 0..9, i.e. every
//     type except King) keep their layout at 64-wide stride, just renumbered into
//     a 704-wide-per-bucket table (bucket * 704 + p_idx * 64 + oriented_sq).
//     The two King planes (own p_idx=10, enemy p_idx=11) are merged into a
//     single 64-wide block at p_idx'=10: the enemy king contributes its actual
//     oriented square, and the own king contributes exactly one fixed slot
//     (oriented_sq == oriented king square, since the anchor king's own square
//     *is* the king square by construction) -- so in real gameplay this
//     collapses to "treat King as p_idx'=10 regardless of which side it is",
//     with own/enemy king naturally landing on different squares.

#include <cstdint>

#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"

namespace nnue::halfka
{
    constexpr int NUM_SQ = 64;
    constexpr int NUM_REAL_PLANES = 704; // 11 piece "slots" * 64 squares
    constexpr int NUM_BUCKETS = 32;
    constexpr int NUM_REAL_FEATURES = NUM_REAL_PLANES * NUM_BUCKETS; // 22,528

    // clang-format off
    constexpr int KingBuckets[64] = {
      -1, -1, -1, -1, 31, 30, 29, 28,
      -1, -1, -1, -1, 27, 26, 25, 24,
      -1, -1, -1, -1, 23, 22, 21, 20,
      -1, -1, -1, -1, 19, 18, 17, 16,
      -1, -1, -1, -1, 15, 14, 13, 12,
      -1, -1, -1, -1, 11, 10, 9, 8,
      -1, -1, -1, -1, 7, 6, 5, 4,
      -1, -1, -1, -1, 3, 2, 1, 0
    };
    // clang-format on

    template <Color perspective>
    inline int orient(int sq, int ksq)
    {
        const int kfile = ksq % 8;
        int x = sq;
        if (kfile < 4)
            x ^= 7;
        if constexpr (perspective == BLACK)
            x ^= 56;
        return x;
    }

    // Returns the real (11-piece-type, 22,528-wide) HalfKAv2_hm feature index
    // for a single piece, from the given perspective. `king_sq` is this
    // perspective's own king square (unoriented, i.e. plain 0..63 Square).
    template <Color perspective>
    inline int feature_index(int king_sq, Color piece_color, Piece piece_type, int piece_sq)
    {
        const int o_ksq = orient<perspective>(king_sq, king_sq);
        const int bucket = KingBuckets[o_ksq];

        const int p_idx = (piece_type == KING)
                               ? 10
                               : static_cast<int>(piece_type) * 2 + (piece_color != perspective ? 1 : 0);

        const int o_sq = orient<perspective>(piece_sq, king_sq);

        return bucket * NUM_REAL_PLANES + p_idx * NUM_SQ + o_sq;
    }
}
