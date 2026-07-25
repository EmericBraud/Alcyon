#pragma once

// Tuning constants for the NNUE lazy-materialization machinery (see
// NnueEval::materialize in nnue_eval.hpp).
namespace engine_constants::nnue
{
    // When a full eval has to catch the L1 accumulator up by at least this
    // many buffered plies at once, it is rebuilt from scratch off the
    // current board (one depth-tagged snapshot + a full two-perspective
    // scan) instead of replaying every per-ply diff (one ~4KB snapshot plus
    // a filtered weight-row diff per ply). Determined empirically via
    // `bench 12` sweeps -- see the perf notes/memory for the measurements.
    constexpr int AccRebuildMinPlies = 1000000;

    // Same cutover when at least one of the buffered plies is a king move
    // (PendingDiff::refresh): such a ply already forces a reset + full
    // activation replay of the mover's perspective at apply time, so the
    // from-scratch rebuild pays for itself earlier.
    constexpr int AccRebuildKingMinPlies = 2;
}
