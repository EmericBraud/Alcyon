#include "gpu_queue.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <vector>

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
    std::vector<float> planes_batch;
    std::vector<int> piece_counts;
    std::vector<std::uint64_t> child_keys;

    while (running_.load(std::memory_order_relaxed)) {
        std::optional<GpuTask> task = pop_one(tasks_, lock_, tail_, count_);
        if (!task) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        if (!scratch.load_fen(task->position.to_fen())) {
            continue; // shouldn't happen -- defensively skip a malformed snapshot
        }

        planes_batch.clear();
        piece_counts.clear();
        child_keys.clear();

        for (int i = 0; i < task->num_candidates; ++i) {
            const Move move = task->candidate_moves[i];
            if (!scratch.is_move_pseudo_legal(move) || !scratch.is_move_legal(move)) {
                continue; // stale by the time the GPU thread gets to it -- skip
            }
            scratch.play(move);

            const GpuPosition child = GpuPosition::from_board(scratch);
            float planes[kNumPlanesV3][kPlaneSize];
            encode_planes_v3(child, planes);
            // Debug aid: set CHESS26_GPU_DEBUG_DUMP_PLANES=/path/to/file to
            // dump the last-encoded position's raw plane tensor (+ FEN in
            // a sibling .fen file) for cross-checking against
            // training/cnn/eval_compare's Python encoder on the exact same
            // FEN -- e.g. how the plane-encoder/Metal-inference numerical
            // parity was verified (byte-identical planes, matching score)
            // during development. No-op (one getenv call) when unset.
            if (const char *dump_path = std::getenv("CHESS26_GPU_DEBUG_DUMP_PLANES")) {
                std::ofstream out(dump_path, std::ios::binary);
                out.write(reinterpret_cast<const char *>(planes), sizeof(planes));
                std::ofstream fen_out(std::string(dump_path) + ".fen");
                fen_out << child.to_fen() << " piece_count=" << non_king_piece_count(child);
            }
            planes_batch.insert(planes_batch.end(), &planes[0][0], &planes[0][0] + kNumPlanesV3 * kPlaneSize);
            piece_counts.push_back(non_king_piece_count(child));
            child_keys.push_back(child.zobrist_key);

            scratch.unplay(move);
        }

        const int batch_size = static_cast<int>(child_keys.size());
        if (batch_size == 0) {
            continue;
        }

        std::vector<std::int32_t> scores(static_cast<std::size_t>(batch_size));
        GpuBackend::instance().infer_batch(planes_batch.data(), piece_counts.data(), batch_size, scores.data());

        for (int i = 0; i < batch_size; ++i) {
            shared_gpu_tt().store(child_keys[static_cast<std::size_t>(i)],
                                   static_cast<std::int16_t>(scores[static_cast<std::size_t>(i)]),
                                   task->depth);
        }
    }
}

GpuQueue &shared_gpu_queue() {
    static GpuQueue instance;
    return instance;
}

} // namespace gpu_eval
