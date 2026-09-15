// Thin C++ wrapper over the extern "C" boundary declared in
// gpu_backend.hpp -- compiled by the project's normal C++ compiler
// (always part of chess_core), never by Apple Clang. See that header's
// comment for why this split exists.
//
// Also the runtime switch between the two implementations (Neural Engine
// via CoreML, GPU via MPSGraph). Plain relaxed atomic rather than a mutex:
// selection changes only from the UCI thread, between searches, while
// infer_batch() reads it from the GPU-prep thread.
#include "gpu_backend.hpp"

#include <atomic>

namespace gpu_eval {

namespace {
std::atomic<Backend> g_active{Backend::Ane};
} // namespace

GpuBackend &GpuBackend::instance() {
    static GpuBackend backend;
    return backend;
}

void GpuBackend::select(Backend backend) { g_active.store(backend, std::memory_order_relaxed); }

Backend GpuBackend::active() const { return g_active.load(std::memory_order_relaxed); }

bool GpuBackend::load_weights(const std::string &weights_path) {
    return active() == Backend::Ane ? chess26_gpu_ane_load_weights(weights_path.c_str())
                                    : chess26_gpu_metal_load_weights(weights_path.c_str());
}

bool GpuBackend::is_ready() const {
    return active() == Backend::Ane ? chess26_gpu_ane_is_ready() : chess26_gpu_metal_is_ready();
}

void GpuBackend::infer_batch(const float *planes_batch, const int *piece_counts, int batch_size,
                              std::int32_t *out_scores_cp) {
    if (active() == Backend::Ane) {
        chess26_gpu_ane_infer_batch(planes_batch, piece_counts, batch_size, out_scores_cp);
    } else {
        chess26_gpu_metal_infer_batch(planes_batch, piece_counts, batch_size, out_scores_cp);
    }
}

} // namespace gpu_eval
