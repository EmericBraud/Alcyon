#pragma once

#include <cstdint>
#include <vector>

#include "chess.h"
#include "training_data_entry.h"

namespace chess26::cnn {

// Must match the plane layout documented in training/cnn/model.py
// (NUM_PLANES and the comment above it) — keep the two in sync.
constexpr int NUM_PLANES = 21;
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
    // Binary attack maps: square is attacked by at least one of our/
    // their pieces (pseudo-legal attacks, king included, same notion
    // as Board::isSquareAttacked). Gives the trunk a directly usable
    // signal for piece mobility/reach that would otherwise take
    // several conv layers to approximate (a rook's attack ray spans
    // the whole board, a 3x3 kernel does not).
    PLANE_US_ATTACKS = 19,
    PLANE_THEM_ATTACKS = 20,
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
