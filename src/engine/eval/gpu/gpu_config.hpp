#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// GPU-eval subsystem: a dedicated CPU thread encodes leaf-of-PV positions
// (per search worker) into CNN input planes and batches them to a Metal
// CNN (v3 architecture, see training/cnn/legacy_model.py), writing scores
// into a small separate TT (see gpu_tt.hpp) that the main TT consults on
// probe. See docs/gpu-async-eval/architecture.md for the design rationale.
//
// CHESS26_GPU_EVAL_METAL is defined by CMake only when ENABLE_GPU_EVAL is
// on AND the target is Apple (Metal is the only backend for now). All of
// this module's headers compile unconditionally on every platform -- the
// Metal-specific implementation lives solely in metal_backend.mm, with a
// no-op stub (metal_backend_stub.cpp) built instead when the macro is
// absent, so callers (worker.cpp, transp_table.hpp) never need #ifdefs.
namespace gpu_eval {

// Runtime on/off switch -- toggled by the "setoption name gpueval value
// true/false" UCI command (see uci.hpp). Checked (relaxed) on the search
// hot path before doing any GPU-eval work, so leaving it false costs one
// atomic load. Also gates whether the dedicated thread is even started.
inline std::atomic<bool> enabled{false};

// How many candidate replies to encode and batch per PV-leaf position.
// Tunable later; kept as a compile-time constant for now so the fixed
// -size buffers below don't need runtime resizing.
inline constexpr int kNumCandidateMoves = 5;

// Root-move-improvement submissions (see negamax.cpp's ply==0 loop and
// SearchWorker::maybe_submit_pv_leaf_to_gpu_throttled) happen on the
// search hot path, unlike the once-per-completed-depth call from
// iterative_deepening() (off the hot path, always unthrottled) -- so
// they're gated by a minimum depth (skip shallow nodes, where the
// root's best move changes often and cheaply, not worth the move-gen +
// scoring cost). A node-count throttle was tried and removed after
// measurement showed it wasn't needed: GpuQueue::push() already drops
// silently on a full/contended queue, so bursty submissions just get
// dropped rather than causing unbounded cost.
inline constexpr int kMinDepthForMidSearchSubmit = 6;

// How many queued PV-leaf tasks the GPU thread drains and encodes
// together before firing a single GpuBackend::infer_batch() call (see
// GpuQueue::run(), gpu_queue.cpp). A small network like this one is
// dominated by MPSGraph's per-call launch overhead rather than raw
// compute, so batching several tasks' candidates into one Metal call
// amortizes that overhead instead of paying it once per 5-candidate
// task. kMaxBatchPositions bounds the fixed-size encoding buffers (no
// runtime allocation on this thread's hot loop).
inline constexpr int kMaxDrainTasksPerBatch = 8;
inline constexpr int kMaxBatchPositions = kMaxDrainTasksPerBatch * kNumCandidateMoves;

// Fixed-capacity ring buffer for pending GPU-eval tasks (see gpu_queue.hpp).
// Must be a power of two (index masking, no modulo). One task per
// worker-iteration PV leaf, so this only needs to cover "a few pending
// iterations behind the dedicated thread", not one slot per search node.
inline constexpr std::size_t kQueueCapacity = 256;

// Preallocated size of the small GPU-score cache (gpu_tt.hpp). 4 MB for
// now, matching a single 32-byte(ish) entry per bucket at a modest entry
// count -- revisit once the batching pipeline is measured end-to-end.
inline constexpr std::size_t kGpuTTSizeBytes = 4ull * 1024 * 1024;

} // namespace gpu_eval
