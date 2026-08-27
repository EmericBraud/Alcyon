#pragma once

#include <cstdint>
#include <vector>

#include "chess.h"
#include "training_data_entry.h"

namespace chess26::cnn {

// Must match the plane layout documented in training/cnn/model.py
// (NUM_PLANES and the comment above it) — keep the two in sync.
constexpr int NUM_PLANES = 33;
constexpr int BOARD_SIZE = 8;
constexpr int PLANE_SIZE = BOARD_SIZE * BOARD_SIZE;

// Plane indices, from the perspective of the side to move ("us").
enum PlaneIndex : int {
    PLANE_US_PAWN = 0,
    PLANE_US_KNIGHT = 1,
    PLANE_US_BISHOP = 2,
    PLANE_US_ROOK = 3,
    PLANE_US_QUEEN = 4,
    PLANE_US_KING = 5,
    PLANE_THEM_PAWN = 6,
    PLANE_THEM_KNIGHT = 7,
    PLANE_THEM_BISHOP = 8,
    PLANE_THEM_ROOK = 9,
    PLANE_THEM_QUEEN = 10,
    PLANE_THEM_KING = 11,
    PLANE_SIDE_TO_MOVE_IS_WHITE = 12,
    PLANE_US_CASTLE_KINGSIDE = 13,
    PLANE_US_CASTLE_QUEENSIDE = 14,
    PLANE_THEM_CASTLE_KINGSIDE = 15,
    PLANE_THEM_CASTLE_QUEENSIDE = 16,
    PLANE_EN_PASSANT = 17,
    PLANE_RULE50 = 18,
    // Binary attack maps, one per piece type per side (pseudo-legal
    // attacks, same notion as Board::isSquareAttacked but broken down
    // by attacker type instead of aggregated) — lets the trunk tell
    // "attacked by a pawn" from "attacked by a queen" directly,
    // rather than having to infer it by combining an aggregated
    // attack plane with piece-position planes across several conv
    // layers (a 1x1 conv can't do it — the attacker's position and
    // the attacked square are different squares by definition).
    // Order matches PieceType ordinal (Pawn..King), offset by 6 for
    // "them" — mirrors PLANE_US_PAWN/PLANE_THEM_PAWN above.
    PLANE_US_PAWN_ATTACKS = 19,
    PLANE_US_KNIGHT_ATTACKS = 20,
    PLANE_US_BISHOP_ATTACKS = 21,
    PLANE_US_ROOK_ATTACKS = 22,
    PLANE_US_QUEEN_ATTACKS = 23,
    PLANE_US_KING_ATTACKS = 24,
    PLANE_THEM_PAWN_ATTACKS = 25,
    PLANE_THEM_KNIGHT_ATTACKS = 26,
    PLANE_THEM_BISHOP_ATTACKS = 27,
    PLANE_THEM_ROOK_ATTACKS = 28,
    PLANE_THEM_QUEEN_ATTACKS = 29,
    PLANE_THEM_KING_ATTACKS = 30,
    // Chebyshev distance (in squares) from each square to our/their king,
    // normalized to [0, 1] (max possible distance on an 8x8 board is 7).
    // A continuous per-square "field" centered on the king, giving the
    // trunk a direct positional signal for king-safety-relevant distance
    // without needing several conv layers to triangulate it from the raw
    // king-position plane alone — mirrors the role HalfKA's king-relative
    // feature indexing plays for the NNUE, without replicating its full
    // per-king-square bucketing (see docs/gpu-async-eval/v4-ideas.md).
    PLANE_US_KING_DISTANCE = 31,
    PLANE_THEM_KING_DISTANCE = 32,
};

// Dense CNN input batch, built directly from binpack::TrainingDataEntry
// (bypassing the NNUE sparse feature extractors entirely). Mirrors the
// public-field-block layout of SparseBatch/FenBatch in nnue-pytorch so
// the ctypes wrapper on the Python side follows the same pattern.
struct PlaneBatch final {
    explicit PlaneBatch(const std::vector<binpack::TrainingDataEntry>& entries);
    ~PlaneBatch();

    PlaneBatch(const PlaneBatch&) = delete;
    PlaneBatch& operator=(const PlaneBatch&) = delete;

    int size;

    // size * NUM_PLANES * PLANE_SIZE, row-major (batch, plane, rank, file).
    float* planes;

    // size. Raw binpack score (from the side to move's perspective,
    // same convention as TrainingDataEntry::score), not yet converted
    // to WDL — that conversion happens in the Python training script
    // once cp_scale is calibrated (see model.py).
    float* score;

    // size. Game result from the side to move's perspective: +1 win,
    // 0 draw, -1 loss (same convention as TrainingDataEntry::result).
    float* result;

    // size. Total non-king pieces on the board (both sides). Free to
    // compute here since we already walk all 64 squares — used on
    // the Python side to pick which value head to route each sample
    // to (see model.py's phase-bucketed heads). Kings excluded since
    // they're always exactly 2 and carry no phase information.
    int* piece_count;

private:
    void fill_entry(int i, const binpack::TrainingDataEntry& e);
};

// Deterministic 64-bit hash of a position, independent of thread/
// process/read order — used to split a single binpack file into
// disjoint, reproducible train/validation subsets by position rather
// than by physically cutting the file (which would corrupt its
// delta-compressed per-game encoding, see binpack.h). Same position
// always hashes the same way, so train and validation streams reading
// the same file with complementary predicates never overlap.
std::uint64_t hash_position(const chess::Position& pos);

}  // namespace chess26::cnn
