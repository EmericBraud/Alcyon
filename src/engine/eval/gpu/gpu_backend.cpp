// Thin C++ wrapper over the extern "C" boundary declared in
// gpu_backend.hpp -- compiled by the project's normal C++ compiler
// (always part of chess_core), never by Apple Clang. See that header's
// comment for why this split exists.
#include "gpu_backend.hpp"

namespace gpu_eval {

GpuBackend &GpuBackend::instance() {
    static GpuBackend backend;
    return backend;
}

bool GpuBackend::load_weights(const std::string &weights_path) {
    return chess26_gpu_backend_load_weights(weights_path.c_str());
}

bool GpuBackend::is_ready() const { return chess26_gpu_backend_is_ready(); }

void GpuBackend::infer_batch(const float *planes_batch, const int *piece_counts, int batch_size,
                              std::int32_t *out_scores_cp) {
    chess26_gpu_backend_infer_batch(planes_batch, piece_counts, batch_size, out_scores_cp);
}

} // namespace gpu_eval
