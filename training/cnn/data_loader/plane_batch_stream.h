#pragma once

#include <memory>
#include <string>
#include <vector>

#include "nnue_training_data_stream.h"
#include "plane_batch.h"

namespace chess26::cnn {

// v1: synchronous, single-threaded fetch — no background prefetch
// worker yet. SparseBatch/FenBatch in nnue-pytorch use a threaded
// producer/consumer queue (see FeaturedBatchStream) for throughput;
// add the same pattern here once this path is validated end-to-end
// and prefetch is confirmed to be a bottleneck for CNN training.
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
    PlaneBatchStream(int concurrency,
                      const std::vector<std::string>& filenames,
                      int batch_size,
                      bool cyclic,
                      int val_percent = 0,
                      bool is_validation = false);

    // Returns nullptr once the underlying stream is exhausted and
    // not cyclic. Caller owns the returned batch (delete when done).
    PlaneBatch* next();

private:
    int m_batch_size;
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

}  // namespace chess26::cnn
