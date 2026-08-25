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
    PlaneBatchStream(int concurrency,
                      const std::vector<std::string>& filenames,
                      int batch_size,
                      bool cyclic);

    // Returns nullptr once the underlying stream is exhausted and
    // not cyclic. Caller owns the returned batch (delete when done).
    PlaneBatch* next();

private:
    int m_batch_size;
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

}  // namespace chess26::cnn
