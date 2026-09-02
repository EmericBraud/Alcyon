#include "gpu_queue.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>

#include "core/board/board.hpp"
#include "gpu_backend.hpp"
#include "gpu_encoder.hpp"
#include "gpu_tt.hpp"

#ifdef __APPLE__
#include <pthread.h>
#endif

namespace gpu_eval {

namespace {

// Best-effort: bias the scheduler away from performance cores. macOS
// exposes no public API to pin a thread to a specific physical core
// (unlike Linux's pthread_setaffinity_np, see the commented-out attempt
// in engine_manager.hpp) -- QoS class is the closest available lever,
// and this subsystem is Apple/Metal-only for now (see CMakeLists.txt's
// ENABLE_GPU_EVAL), so that's the only platform this needs to handle.
void pin_to_background_core() {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
}

} // namespace

bool GpuQueue::push(const GpuTask &task) {
    if (lock_.test_and_set(std::memory_order_acquire)) {
        return false; // contended -- drop rather than block a search worker
    }
    bool pushed = false;
    if (count_ < kQueueCapacity) {
        tasks_[head_] = task;
        head_ = (head_ + 1) & kMask;
        ++count_;
        pushed = true;
    }
    lock_.clear(std::memory_order_release);
    return pushed;
}

namespace {
std::optional<GpuTask> pop_one(GpuTask *tasks, std::atomic_flag &lock, std::size_t &tail, std::size_t &count) {
    if (lock.test_and_set(std::memory_order_acquire)) {
        return std::nullopt;
    }
    std::optional<GpuTask> out;
    if (count > 0) {
        out = tasks[tail];
        tail = (tail + 1) & (kQueueCapacity - 1);
        --count;
    }
    lock.clear(std::memory_order_release);
    return out;
}
} // namespace

bool GpuQueue::start(const std::string &weights_path) {
    if (running_.load(std::memory_order_relaxed)) {
        return true; // already running
    }
    // Join a previous attempt's thread first -- a joinable std::thread
    // left dangling at process exit calls std::terminate() -- fatal in
    // this -fno-exceptions build, not a graceful fallback.
    if (thread_.joinable()) {
        thread_.join();
    }

    // Load synchronously, on the CALLER's thread (the UCI command
    // thread), so this function can report success/failure before
    // returning -- avoids a race where a caller checks GpuBackend::
    // is_ready() right after start() returns, before a background
    // thread would have had a chance to even begin loading.
    if (!GpuBackend::instance().load_weights(weights_path)) {
        std::fprintf(stderr, "gpu_eval: failed to load weights from %s, GPU eval disabled\n", weights_path.c_str());
        return false;
    }

    running_.store(true, std::memory_order_relaxed);
    thread_ = std::thread(&GpuQueue::run, this);
    return true;
}

void GpuQueue::stop() {
    running_.store(false, std::memory_order_relaxed);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void GpuQueue::run() {
    pin_to_background_core();

    Board scratch;

    // Fixed-size, allocated once for this thread's whole lifetime -- no
    // runtime allocation in the drain/encode/infer loop below. Sized for
    // kMaxDrainTasksPerBatch tasks' worth of candidates (see
    // gpu_config.hpp for the batching rationale).
    static float planes_batch[kMaxBatchPositions][kNumPlanesV3][kPlaneSize];
    static int piece_counts[kMaxBatchPositions];
    static std::uint64_t child_keys[kMaxBatchPositions];
    static std::uint8_t child_depths[kMaxBatchPositions];
    static std::int32_t scores[kMaxBatchPositions];

    while (running_.load(std::memory_order_relaxed)) {
        int batch_size = 0;

        // Drain up to kMaxDrainTasksPerBatch queued tasks (or until the
        // queue is empty) into the shared buffers above before firing a
        // single infer_batch() call, instead of one call per task.
        int tasks_drained = 0;
        while (tasks_drained < kMaxDrainTasksPerBatch && batch_size + kNumCandidateMoves <= kMaxBatchPositions) {
            std::optional<GpuTask> task = pop_one(tasks_, lock_, tail_, count_);
            if (!task) {
                break; // queue empty for now
            }
            ++tasks_drained;

            if (!scratch.load_fen(task->position.to_fen())) {
                continue; // shouldn't happen -- defensively skip a malformed snapshot
            }

            for (int i = 0; i < task->num_candidates; ++i) {
                const Move move = task->candidate_moves[i];
                if (!scratch.is_move_pseudo_legal(move) || !scratch.is_move_legal(move)) {
                    continue; // stale by the time the GPU thread gets to it -- skip
                }
                scratch.play(move);

                // Skip re-encoding + re-inferring a candidate whose score
                // is already fresh in the GPU TT (same key, current
                // generation) -- measured (via GpuTT::redundant_stores())
                // at ~60-70% of all candidates in a typical search, since
                // neighboring PV-leaf submissions often share the same
                // downstream child positions. Checking here (before
                // encode_planes_v3, the actual compute-heavy step) avoids
                // that wasted work instead of just detecting it after the
                // fact in store().
                std::int16_t existing_score;
                std::uint8_t existing_depth, existing_age;
                const std::uint64_t child_key = scratch.get_hash();
                if (shared_gpu_tt().probe(child_key, existing_score, existing_depth, existing_age) &&
                    existing_age == shared_gpu_tt().current_age()) {
                    scratch.unplay(move);
                    continue;
                }

                // Also dedupe against candidates already added to THIS
                // batch (from this or an earlier drained task) -- these
                // can't be caught by the probe() above since nothing gets
                // store()'d until after the whole batch's infer_batch()
                // call returns, below.
                bool duplicate_in_batch = false;
                for (int j = 0; j < batch_size; ++j) {
                    if (child_keys[j] == child_key) {
                        duplicate_in_batch = true;
                        break;
                    }
                }
                if (duplicate_in_batch) {
                    scratch.unplay(move);
                    continue;
                }

                const GpuPosition child = GpuPosition::from_board(scratch);
                encode_planes_v3(child, planes_batch[batch_size]);
                // Debug aid: set CHESS26_GPU_DEBUG_DUMP_PLANES=/path/to/file to
                // dump the last-encoded position's raw plane tensor (+ FEN in
                // a sibling .fen file) for cross-checking against
                // training/cnn/eval_compare's Python encoder on the exact same
                // FEN -- e.g. how the plane-encoder/Metal-inference numerical
                // parity was verified (byte-identical planes, matching score)
                // during development. No-op (one getenv call) when unset.
                if (const char *dump_path = std::getenv("CHESS26_GPU_DEBUG_DUMP_PLANES")) {
                    std::ofstream out(dump_path, std::ios::binary);
                    out.write(reinterpret_cast<const char *>(planes_batch[batch_size]), sizeof(planes_batch[batch_size]));
                    std::ofstream fen_out(std::string(dump_path) + ".fen");
                    fen_out << child.to_fen() << " piece_count=" << non_king_piece_count(child);
                }
                piece_counts[batch_size] = non_king_piece_count(child);
                child_keys[batch_size] = child.zobrist_key;
                child_depths[batch_size] = task->depth;
                ++batch_size;

                scratch.unplay(move);
            }
        }

        if (batch_size == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        GpuBackend::instance().infer_batch(&planes_batch[0][0][0], piece_counts, batch_size, scores);

        for (int i = 0; i < batch_size; ++i) {
            shared_gpu_tt().store(child_keys[i], static_cast<std::int16_t>(scores[i]), child_depths[i]);
        }
    }
}

GpuQueue &shared_gpu_queue() {
    static GpuQueue instance;
    return instance;
}

} // namespace gpu_eval
