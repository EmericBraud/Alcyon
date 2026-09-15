// No-op backends, built instead of metal_backend.mm/coreml_backend.mm when
// CHESS26_GPU_EVAL_METAL is not defined (non-Apple builds, or Metal
// support disabled) -- see CMakeLists.txt. Keeps gpu_queue.cpp free of
// platform #ifdefs: with the GPU eval subsystem disabled at the CMake
// level, load fails and is_ready is always false, so gpu_eval::enabled
// should never be set true by the UCI handler in that build (see
// uci.hpp) -- but this stub exists as a safety net regardless.
#include "gpu_backend.hpp"

bool chess26_gpu_metal_load_weights(const char * /*weights_path*/) { return false; }
bool chess26_gpu_metal_is_ready() { return false; }
void chess26_gpu_metal_infer_batch(const float * /*planes_batch*/, const int * /*piece_counts*/, int /*batch_size*/,
                                    std::int32_t * /*out_scores_cp*/) {}

bool chess26_gpu_ane_load_weights(const char * /*model_path*/) { return false; }
bool chess26_gpu_ane_is_ready() { return false; }
void chess26_gpu_ane_infer_batch(const float * /*planes_batch*/, const int * /*piece_counts*/, int /*batch_size*/,
                                  std::int32_t * /*out_scores_cp*/) {
    // Unreachable in practice (is_ready() is false), left empty rather
    // than asserting so a misuse here degrades silently instead of
    // crashing a running search.
}
