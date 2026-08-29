#pragma once

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "nnue_training_data_stream.h"
#include "plane_batch.h"

namespace chess26::cnn {

// Synchronous, single-threaded core: blocks on the underlying binpack
// reader for every batch. Used only as the producer inside
// PlaneBatchStream's background thread (below) — not exposed via the
// ABI directly, since callers always want the prefetching version.
class PlaneBatchStreamImpl {
public:
    // rank/world_size: for multi-GPU (DDP) training, shards the
    // underlying binpack file(s) across processes -- each rank reads a
    // disjoint chunk sequence (see binpack.h's
    // CompressedTrainingDataEntryParallelReader "DDP seeking" logic),
    // so world_size streams built from the same files with different
    // ranks never see the same chunk of positions. rank=0/world_size=1
    // (the defaults) disables sharding entirely -- every position goes
    // to the single stream, as before.
    PlaneBatchStreamImpl(int concurrency,
                          const std::vector<std::string>& filenames,
                          int batch_size,
                          bool cyclic,
                          int val_percent,
                          bool is_validation,
                          std::string nnue_path,
                          int rank,
                          int world_size);

    // Returns nullptr once the underlying stream is exhausted and not
    // cyclic. Caller owns the returned batch.
    PlaneBatch* next();

private:
    int m_batch_size;
    std::string m_nnue_path;
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

// Prefetching wrapper: a background thread keeps calling
// PlaneBatchStreamImpl::next() and pushes completed batches into a
// bounded queue, so the consumer (the training loop, blocked on GPU
// compute or Python-side tensor conversion) finds a batch already
// waiting instead of blocking on binpack I/O + plane encoding for
// every step. This is what closed the CPU-bound gap identified when
// profiling training on a single synchronous thread (see
// docs/gpu-async-eval/ discussion) — the model here is small enough
// that GPU compute alone rarely keeps up with a purely synchronous
// loader, so the queue buys back the idle time on both sides.
class PlaneBatchStream {
public:
    // val_percent: 0-100, the share of positions (by hash_position,
    // see plane_batch.h) routed to the validation split.
    // is_validation: false -> stream yields the (100 - val_percent)%
    // training positions; true -> yields the val_percent% validation
    // positions. Two streams built from the same filenames with
    // is_validation flipped are disjoint and reproducible — no
    // physical file split needed. Pass val_percent=0 to disable
    // splitting entirely (every position goes to the training/only
    // stream), e.g. when validation comes from a separate file.
    // queue_capacity: max number of fully-built PlaneBatch objects
    // held in memory ahead of consumption. Higher hides more I/O
    // latency but costs more RAM (each batch is
    // batch_size * NUM_PLANES * 64 floats, see plane_batch.h).
    // rank/world_size: see PlaneBatchStreamImpl's constructor doc above.
    PlaneBatchStream(int concurrency,
                      const std::vector<std::string>& filenames,
                      int batch_size,
                      bool cyclic,
                      int val_percent = 0,
                      bool is_validation = false,
                      std::string nnue_path = "",
                      int rank = 0,
                      int world_size = 1,
                      int queue_capacity = 4);
    ~PlaneBatchStream();

    PlaneBatchStream(const PlaneBatchStream&) = delete;
    PlaneBatchStream& operator=(const PlaneBatchStream&) = delete;

    // Returns nullptr once the underlying stream is exhausted and not
    // cyclic. Caller owns the returned batch (delete when done).
    PlaneBatch* next();

private:
    void worker_loop();

    PlaneBatchStreamImpl m_impl;
    std::size_t m_capacity;

    std::mutex m_mutex;
    std::condition_variable m_not_empty;
    std::condition_variable m_not_full;
    std::deque<PlaneBatch*> m_queue;
    bool m_exhausted = false;
    bool m_stop = false;

    std::thread m_worker;
};

}  // namespace chess26::cnn
