#include "plane_batch_abi.h"

#include <string>
#include <vector>

#include "plane_batch.h"
#include "plane_batch_stream.h"

using chess26::cnn::PlaneBatch;
using chess26::cnn::PlaneBatchStream;

struct PlaneBatchCStream {
    PlaneBatchStream impl;
};

PLANE_API PlaneBatchCStream* create_plane_batch_stream(int concurrency,
                                                         int num_files,
                                                         const char* const* filenames,
                                                         int batch_size,
                                                         bool cyclic,
                                                         int val_percent,
                                                         bool is_validation,
                                                         const char* nnue_path,
                                                         int rank,
                                                         int world_size) {
    std::vector<std::string> filenames_vec(filenames, filenames + num_files);
    return new PlaneBatchCStream{
        PlaneBatchStream(concurrency, filenames_vec, batch_size, cyclic, val_percent, is_validation,
                          std::string(nnue_path), rank, world_size)};
}

PLANE_API void destroy_plane_batch_stream(PlaneBatchCStream* stream) { delete stream; }

PLANE_API PlaneBatchCView fetch_next_plane_batch(PlaneBatchCStream* stream) {
    PlaneBatch* batch = stream->impl.next();
    if (batch == nullptr) {
        return PlaneBatchCView{0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return PlaneBatchCView{batch->size, batch->planes, batch->score, batch->result,
                            batch->piece_count, batch->nnue_score, batch};
}

PLANE_API void destroy_plane_batch(PlaneBatchCView view) {
    delete static_cast<PlaneBatch*>(view.handle);
}
