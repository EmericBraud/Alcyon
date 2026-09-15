#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

#include "core/move/move.hpp"
#include "gpu_config.hpp"
#include "gpu_position.hpp"

// The one dedicated GPU-prep thread: pops PV-leaf tasks pushed by search
// workers (see SearchWorker::maybe_submit_pv_leaf_to_gpu, worker.cpp),
// applies each task's candidate moves (via Board::play(), see
// gpu_queue.cpp) to rebuild each resulting position, batches their
// encoded planes (gpu_encoder.hpp) through GpuBackend, and writes scores
// into shared_gpu_tt() (gpu_tt.hpp) for the main TT to consult on probe.
//
// Non-blocking producer side: push() takes a spinlock with try_lock only
// -- a search worker that can't immediately acquire it (or finds the
// queue full) just drops the task rather than waiting, since a dropped
// PV-leaf submission costs nothing but a missed cache-warm opportunity.
namespace gpu_eval {

struct GpuTask {
    GpuPosition position;
    Move candidate_moves[kNumCandidateMoves];
    int num_candidates = 0;
    std::uint8_t depth = 0; // search depth (plies) this PV leaf was captured at

    // Phase 0 only (measure_diagnostics()), both filled by
    // SearchWorker::submit_current_position_to_gpu:
    //
    //  tt_move     -- the move the search itself settled on at this position
    //                 (its main-TT entry). The TARGET both predictors are
    //                 scored against.
    //  blind_best  -- the best candidate according to the move-ordering
    //                 heuristics ALONE, re-ranked with tt_move suppressed.
    //                 Without that suppression the comparison is circular:
    //                 score_move gives tt_move a 9600 bonus, so our first
    //                 pick would BE tt_move by construction and agree 100%.
    //                 The CNN is likewise blind to tt_move, so the two
    //                 predictors face the target on equal terms.
    Move tt_move = 0;
    Move blind_best = 0;
};

class GpuQueue {
public:
    // Non-blocking: returns false (task dropped) if the queue is full or
    // momentarily contended. Safe to call from any search worker thread.
    bool push(const GpuTask &task);

    // Loads weights_path into GpuBackend SYNCHRONOUSLY (blocks the
    // calling thread -- the UCI command thread -- until it's done; this
    // only ever happens once, when the "gpueval" UCI option is flipped
    // on, never on the search hot path), then, only on success, spawns
    // the dedicated GPU-prep thread (best-effort pinned to the last
    // logical CPU, see gpu_queue.cpp). Returns whether loading
    // succeeded -- callers must check this (or GpuBackend::is_ready())
    // before setting gpu_eval::enabled, since a failed load leaves no
    // thread running and GpuBackend not ready.
    bool start(const std::string &weights_path);

    // Signals the thread to exit and joins it. Call on engine shutdown.
    void stop();

private:
    void run();

    static constexpr std::size_t kMask = kQueueCapacity - 1;
    static_assert((kQueueCapacity & kMask) == 0, "kQueueCapacity must be a power of two");

    GpuTask tasks_[kQueueCapacity];
    std::atomic_flag lock_ = ATOMIC_FLAG_INIT;
    std::size_t head_ = 0; // next slot to write (producer)
    std::size_t tail_ = 0; // next slot to read (consumer, thread-owned)
    std::size_t count_ = 0;

    std::atomic<bool> running_{false};
    std::thread thread_;
};

GpuQueue &shared_gpu_queue();

} // namespace gpu_eval
