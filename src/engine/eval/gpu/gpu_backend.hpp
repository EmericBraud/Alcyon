#pragma once

#include <cstdint>
#include <string>

#include "gpu_encoder.hpp"

// Backend-agnostic inference interface.
//
// The ACTUAL Metal implementation (metal_backend.mm) is Objective-C++,
// which needs Apple Clang (GCC's Objective-C++ support doesn't accept
// -fobjc-arc at all). The rest of chess_core is compiled with whatever
// C++ compiler the user configured (GCC by default -- see
// accumulator_layer.hpp's use of GCC's <experimental/simd> extension,
// which doesn't exist in Apple Clang's libc++, so building the WHOLE
// engine with Apple Clang isn't currently an option). CMakeLists.txt
// therefore compiles metal_backend.mm with a SEPARATELY-configured
// CMAKE_OBJCXX_COMPILER (forced to Apple Clang) while the rest of the
// project keeps its normal compiler -- meaning object files from two
// different C++ toolchains/standard libraries get linked into one
// binary. That's only safe if NO C++-standard-library type (std::string,
// std::vector, ...) ever crosses the boundary between them, since e.g.
// libstdc++'s std::string layout and mangled name are NOT the same as
// libc++'s.
//
// So the actual cross-compiler boundary is the extern "C" block below
// (POD types only: bool, int32_t, float*, const char*) -- gpu_backend.cpp
// (compiled by the project's normal compiler, part of chess_core as
// always) implements the nice GpuBackend C++ wrapper class in terms of
// these, and is the ONLY thing gpu_queue.cpp/uci.hpp ever call. Only
// metal_backend.mm (Apple-Clang-only) and gpu_backend_stub.cpp (built
// instead when CHESS26_GPU_EVAL_METAL isn't defined) implement the
// extern "C" functions themselves.
extern "C" {
bool chess26_gpu_backend_load_weights(const char *weights_path);
bool chess26_gpu_backend_is_ready();
void chess26_gpu_backend_infer_batch(const float *planes_batch, const int *piece_counts, int batch_size,
                                      std::int32_t *out_scores_cp);
}

namespace gpu_eval {

class GpuBackend {
public:
    static GpuBackend &instance();

    // Loads the raw exported weights (see
    // training/cnn/eval_compare/export_weights_for_metal.py) for v3's
    // architecture (14 SE-residual blocks x 160 channels, 31 input
    // planes, 4 phase-bucket heads, PSQT skip -- see
    // training/cnn/legacy_model.py). Returns false (and leaves the
    // backend unusable) on any failure -- caller must not call
    // infer_batch() until this returns true.
    bool load_weights(const std::string &weights_path);

    bool is_ready() const;

    // planes_batch: batch_size * kNumPlanesV3 * kPlaneSize floats,
    // row-major (batch, plane, square) -- same layout encode_planes_v3()
    // fills per-position. piece_counts: non-king piece count per
    // position (see non_king_piece_count()), used for phase-bucket head
    // selection. out_scores_cp: caller-owned, batch_size ints, receives
    // the win-probability logit converted to centipawns (same
    // sigmoid-family scale as engine NNUE static eval).
    void infer_batch(const float *planes_batch, const int *piece_counts, int batch_size, std::int32_t *out_scores_cp);

private:
    GpuBackend() = default;
};

} // namespace gpu_eval
