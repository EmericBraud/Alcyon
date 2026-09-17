#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

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
namespace gpu_eval
{

    inline std::atomic<bool> enabled{false};

    constexpr inline bool active()
    {
#ifdef CHESS26_GPU_EVAL_METAL
        return enabled.load(std::memory_order_relaxed);
#else
        return false;
#endif
    }

    // How many candidate replies to encode and batch per PV-leaf position.
    // Kept as a compile-time constant so the fixed-size buffers below don't
    // need runtime resizing.
    //
    // The cheapest volume multiplier there is: the submitting search thread
    // already generated and scored EVERY legal move (see
    // SearchWorker::submit_current_position_to_gpu), so taking more of them
    // costs it a few extra pick_best_move() passes and nothing else -- while
    // the GPU thread's real cost is per-INFERENCE-CALL, not per-position
    // (measured ~42ms per call almost regardless of batch size), so a fuller
    // batch is nearly free throughput.
    inline constexpr int kNumCandidateMoves = 4;

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
    inline int mid_search_submit_min_depth()
    {
        static const int v = std::getenv("CHESS26_GPU_MID_DEPTH") ? std::atoi(std::getenv("CHESS26_GPU_MID_DEPTH")) : 3;
        return v;
    }

    // How far apart (in cp) NNUE and the CNN have to be before the two count
    // as "disagreeing" on a position. MEASUREMENT ONLY now (see GpuTT::
    // record_cnn_vs_nnue) -- nothing in the engine acts on the verdict.
    //
    // Two mechanisms built on this were tried and both removed after
    // measurement: gating the score on the search thread (an extra NNUE eval
    // + a bounded search extension per disagreement, net -23 Elo over 300
    // games -- it competed with the rest of the tree for the same time
    // budget), then resolving it on the GPU-prep thread instead (no search
    // cost, but it ate ~90% of that thread's wall clock and replaced the CNN
    // score with a worse 3-ply search on >50% of positions). See
    // docs/gpu-async-eval/consultative-eval-measurements.md.
    inline constexpr int kNeutralAgreementMaxGapCp = 50;

    // Added to every CNN score before it is stored, to put it on the same
    // scale as the search's own eval. Measured (GpuTT::cnn_to_nnue_slope/
    // _intercept_cp/_correlation over four middlegame + endgame searches):
    // the CNN tracks NNUE closely -- slope ~1.0, correlation 0.90-0.94 --
    // but sits a CONSTANT ~165cp more optimistic for the side to move on the
    // settled positions this subsystem scores. Left uncorrected, every
    // stored score injects that optimism straight into the search.
    //
    // ponytail: one global constant, fitted across four positions
    // (per-position intercepts -137..-196cp). If the residual matters, refit
    // per phase bucket (the model already has four of them, see
    // metal_backend.mm) -- cnn_to_nnue_intercept_cp() reading near 0 is the
    // check that this value is still right.
    inline constexpr int kCnnToNnueOffsetCp = -165;

    // Transposition submissions (see negamax.cpp's TT-probe branch and
    // SearchWorker::maybe_submit_transposition_to_gpu): a TT hit proves the
    // position was already reached via a different move order/search path,
    // making it a much better bet for "will this be looked at again" than
    // an arbitrary leaf (tried first: a near-alpha qsearch leaf, measured at
    // ~4% useful_hits -- most are refuted branches alpha-beta abandons for
    // good and never revisits).
    //
    // This is THE volume knob: it used to be 4, which measured at ~60-200
    // positions/s stored -- far too few to be visible in a multi-million-node
    // search (measured ~85 useful hits per 8s search). At 0, with the
    // criticality gate below doing the filtering instead, the same search
    // stores ~3000 positions/s and lands ~1700-1900 useful hits. The GPU
    // device itself is the only thing that should be the limit.
    //
    // WARNING: lowering this multiplies the move-gen + move-scoring work
    // done ON THE SEARCH THREADS (see
    // SearchWorker::submit_current_position_to_gpu). That is exactly the cost
    // pattern that measured -23 Elo in an earlier iteration. Its NPS cost has
    // NOT been measured -- the measurement session ran on battery, where
    // back-to-back identical runs varied by 74% (2.12M vs 3.70M nps) and any
    // NPS comparison is meaningless. Re-measure on AC power before drawing
    // any Elo conclusion from this default.
    inline int transposition_submit_min_depth()
    {
        static const int v = std::getenv("CHESS26_GPU_TT_DEPTH") ? std::atoi(std::getenv("CHESS26_GPU_TT_DEPTH")) : 0;
        return v;
    }

    // How many queued PV-leaf tasks the GPU thread drains and encodes
    // together before firing a single GpuBackend::infer_batch() call (see
    // GpuQueue::run(), gpu_queue.cpp). This network is dominated by
    // MPSGraph's per-call dispatch overhead, not by raw compute -- measured
    // (gpubench): 225 positions/s at batch 1, 4304/s at batch 256, i.e. the
    // cost of a call barely moves with its size. So the batch should be as
    // full as production allows.
    //
    // kMaxBatchPositions bounds the fixed-size encoding buffers (no runtime
    // allocation on this thread's hot loop): 256 * 31 * 64 floats = 2MB,
    // sized to gpubench's measured best throughput point.
    inline constexpr int kMaxBatchPositions = 256;
    // Generously above kMaxBatchPositions / kNumCandidateMoves: most drained
    // candidates are deduped away (measured ~60-78% already fresh in the GPU
    // tt), so draining only enough tasks to nominally fill the buffer leaves
    // it less than half full. The drain loop is bounded by the buffer itself,
    // not by this.
    inline constexpr int kMaxDrainTasksPerBatch = 128;

    // Criticality gate on transposition submissions (see negamax.cpp's
    // TT-probe branch). How near its own cutoff boundary a node's TT score
    // has to be for a refined eval to plausibly change what the node does.
    inline constexpr int kCriticalWindowMarginCp = 100;

    // On by default (opt out with CHESS26_GPU_CRIT=0): with the depth gate
    // above opened to 0, this is what keeps the submissions worth making.
    // Measured via decision_flip_rate_percent(): turning it on raised the
    // share of consumed GPU scores that actually changed a node's outcome at
    // every depth setting tried (48%->57% at depth 0, 49%->53% at 2,
    // 70%->79% at 4).
    inline bool submit_only_critical()
    {
        static const bool on = []
        {
            const char *v = std::getenv("CHESS26_GPU_CRIT");
            return !v || std::atoi(v) != 0;
        }();
        return on;
    }

    // Diagnostic switch (CHESS26_GPU_MEASURE) for everything this subsystem
    // measures about ITSELF rather than needs to run:
    //   - the decision-flip rate at the qsearch consumption site (qsearch.cpp),
    //   - the CNN-vs-NNUE calibration stats on the GPU-prep thread
    //     (gpu_queue.cpp -- GpuTT::record_cnn_vs_nnue and the NNUE eval it
    //     needs).
    // Both cost real work on live paths, and neither feeds a decision: the
    // calibration numbers produced kCnnToNnueOffsetCp once, and re-deriving
    // them every search is only useful when re-tuning it. Off by default;
    // profiling put the NNUE stats eval at 57ms of the prep thread's ~9.5s.
    inline bool measure_diagnostics()
    {
        static const bool on = std::getenv("CHESS26_GPU_MEASURE") != nullptr;
        return on;
    }

    // Fixed-capacity ring buffer for pending GPU-eval tasks (see gpu_queue.hpp).
    // Must be a power of two (index masking, no modulo). Sized so a burst of
    // submissions between two inference calls isn't dropped: at 256 it was
    // measured dropping 16% of pushes while the GPU thread was stalled in a
    // single slow infer_batch() call.
    inline constexpr std::size_t kQueueCapacity = 1024;

    // Preallocated size of the small GPU-score cache (gpu_tt.hpp). 4 MB for
    // now, matching a single 32-byte(ish) entry per bucket at a modest entry
    // count -- revisit once the batching pipeline is measured end-to-end.
    inline constexpr std::size_t kGpuTTSizeBytes = 4ull * 1024 * 1024;

} // namespace gpu_eval
