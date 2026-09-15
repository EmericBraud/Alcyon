// CoreML implementation of the same v3 network the MPSGraph backend runs
// (metal_backend.mm), so the two can be benchmarked against each other on
// the same engine -- see the "gpubench" UCI command.
//
// The point of this path is the Apple Neural Engine: there is no public API
// to target the ANE directly (MPSGraph cannot), CoreML is the only route,
// and even then CoreML DECIDES the placement per layer -- asking for
// MLComputeUnitsCPUAndNeuralEngine is a request, not a guarantee. Verify
// what actually ran where with Xcode's CoreML performance report, or watch
// the block draw power with `sudo powermetrics --samplers ane_power`. A
// silent partial fallback to CPU is the expected failure mode, and it can
// be SLOWER than the GPU backend rather than faster.
//
// The model comes from tools/export_coreml.py, which rebuilds the
// architecture in PyTorch from v3_weights.bin (there is no PyTorch
// definition of it in this repo) and converts it. Two deliberate
// differences from the MPSGraph graph:
//   - Phase-bucket selection is NOT in the model. It outputs all four
//     buckets, [N, 4], and the column is picked here on the CPU -- one array
//     index, and it keeps the model single-input (the ANE handles small side
//     inputs poorly).
//   - Weights are fp16, because that is the only precision the ANE runs.
//   - The model has ONE FIXED batch size (kAneBatch) and larger batches are
//     chunked into it. This is load-bearing: with ct.EnumeratedShapes the ANE
//     silently refused the model outright (CPU_AND_NE measured
//     indistinguishable from CPU_ONLY, 1060 vs 998 positions/s, with an "E5RT
//     encountered an STL exception. msg = std::bad_cast" from the ANE
//     runtime). Fixed-shape, the same network runs at ~15600 positions/s.
#include "gpu_backend.hpp"

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cstdio>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace gpu_eval {

namespace {

// MUST match BATCH_SIZE in tools/export_coreml.py -- the model only
// accepts this exact shape. Anything larger is chunked; a final partial chunk
// is zero-padded and its extra outputs discarded.
//
// 32 is the measured ANE optimum (8 -> 12456, 16 -> 14575, 32 -> 15614,
// 64 -> 14282, 128 -> 13705 positions/s). Note this is the opposite of the
// MPSGraph backend, whose per-call dispatch cost makes it want the biggest
// batch available -- so chunking here is not a compromise: 8 chunks of 32
// beat one call of 256 (14634 vs 12404 positions/s).
constexpr int kAneBatch = 32;

constexpr int kNumPlanes = 31;
constexpr int kPlaneArea = 64;
constexpr int kNumBuckets = 4;

// Same conversion the MPSGraph backend applies (see metal_backend.mm) --
// the network's raw output is a logit, not centipawns.
constexpr float kScoreScale = 410.0f;

// Same bucketing as metal_backend.mm's infer_unsafe().
int bucket_for(int piece_count) {
    static constexpr int kBoundaries[3] = {24, 16, 8};
    int b = 0;
    for (int boundary : kBoundaries) {
        if (piece_count < boundary) {
            ++b;
        }
    }
    return b;
}

} // namespace

class CoreMLImpl {
public:
    bool load(const std::string &model_path) {
        @autoreleasepool {
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:model_path.c_str()]];
            NSError *error = nil;

            // An .mlpackage has to be compiled to an .mlmodelc before it can
            // be loaded. This writes into a temp dir and takes a few seconds
            // for a net this size, which is why it happens here (once, from
            // the UCI thread when gpueval is switched on) and never on the
            // inference path.
            NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
            if (!compiled) {
                std::fprintf(stderr, "gpu_eval(ane): cannot compile %s: %s\n", model_path.c_str(),
                             [[error localizedDescription] UTF8String]);
                return false;
            }

            MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
            // Excludes the GPU on purpose: the GPU path is metal_backend.mm's
            // job, and leaving it available here would let CoreML quietly
            // benchmark the GPU while we believe we are measuring the ANE.
            config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

            model_ = [MLModel modelWithContentsOfURL:compiled configuration:config error:&error];
            if (!model_) {
                std::fprintf(stderr, "gpu_eval(ane): cannot load compiled model: %s\n",
                             [[error localizedDescription] UTF8String]);
                return false;
            }

            // Persistent input array + feature provider, allocated once here
            // rather than per call: the GPU backend's measured per-call floor
            // is dominated by exactly this kind of allocation, so this path
            // should not reintroduce it.
            input_ = [[MLMultiArray alloc] initWithShape:@[ @(kAneBatch), @(kNumPlanes), @8, @8 ]
                                                dataType:MLMultiArrayDataTypeFloat32
                                                   error:&error];
            if (!input_) {
                std::fprintf(stderr, "gpu_eval(ane): cannot allocate input: %s\n",
                             [[error localizedDescription] UTF8String]);
                return false;
            }
            features_ = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"planes" : input_} error:&error];
            if (!features_) {
                std::fprintf(stderr, "gpu_eval(ane): cannot build features: %s\n",
                             [[error localizedDescription] UTF8String]);
                return false;
            }

            ready_ = true;
            return true;
        }
    }

    bool is_ready() const { return ready_; }

    void infer(const float *planes_batch, const int *piece_counts, int batch_size, std::int32_t *out_scores_cp) {
        @autoreleasepool {
            // Same reasoning as metal_backend.mm's infer(): an NSException
            // must never cross the extern "C" boundary into gpu_queue.cpp,
            // which is compiled -fno-exceptions. Fail soft.
            @try {
                infer_unsafe(planes_batch, piece_counts, batch_size, out_scores_cp);
            } @catch (NSException *exception) {
                std::fprintf(stderr, "gpu_eval(ane): NSException during inference: %s -- %s\n",
                             [[exception name] UTF8String], [[exception reason] UTF8String]);
                for (int i = 0; i < batch_size; ++i) {
                    out_scores_cp[i] = 0;
                }
            }
        }
    }

private:
    void infer_unsafe(const float *planes_batch, const int *piece_counts, int batch_size,
                      std::int32_t *out_scores_cp) {
        if (!ready_ || batch_size <= 0) {
            for (int i = 0; i < batch_size; ++i) {
                out_scores_cp[i] = 0;
            }
            return;
        }

        const std::size_t per_position = static_cast<std::size_t>(kNumPlanes) * kPlaneArea;
        for (int base = 0; base < batch_size; base += kAneBatch) {
            const int count = std::min(kAneBatch, batch_size - base);
            float *dst = static_cast<float *>(input_.dataPointer);
            std::memcpy(dst, planes_batch + static_cast<std::size_t>(base) * per_position,
                        static_cast<std::size_t>(count) * per_position * sizeof(float));
            if (count < kAneBatch) {
                // Padding rows still get inferred; their outputs are
                // discarded, but they must not be stale garbage that could be
                // NaN/Inf and poison the shared activations.
                std::memset(dst + static_cast<std::size_t>(count) * per_position, 0,
                            static_cast<std::size_t>(kAneBatch - count) * per_position * sizeof(float));
            }

            NSError *error = nil;
            id<MLFeatureProvider> result = [model_ predictionFromFeatures:features_ error:&error];
            if (!result) {
                std::fprintf(stderr, "gpu_eval(ane): prediction failed: %s\n",
                             [[error localizedDescription] UTF8String]);
                for (int i = base; i < base + count; ++i) {
                    out_scores_cp[i] = 0;
                }
                continue;
            }

            MLMultiArray *logits = [[result featureValueForName:@"bucket_logits"] multiArrayValue];
            const float *values = static_cast<const float *>(logits.dataPointer);
            for (int i = 0; i < count; ++i) {
                const float logit = values[static_cast<std::size_t>(i) * kNumBuckets + bucket_for(piece_counts[base + i])];
                out_scores_cp[base + i] = static_cast<std::int32_t>(logit * kScoreScale);
            }
        }
    }

    MLModel *model_ = nil;
    MLMultiArray *input_ = nil;
    MLDictionaryFeatureProvider *features_ = nil;
    bool ready_ = false;
};

namespace {
CoreMLImpl &impl() {
    static CoreMLImpl instance;
    return instance;
}
} // namespace

} // namespace gpu_eval

// See gpu_backend.hpp: POD-only extern "C" boundary, one set of symbols per
// implementation.
bool chess26_gpu_ane_load_weights(const char *model_path) {
    return gpu_eval::impl().load(std::string(model_path));
}

bool chess26_gpu_ane_is_ready() { return gpu_eval::impl().is_ready(); }

void chess26_gpu_ane_infer_batch(const float *planes_batch, const int *piece_counts, int batch_size,
                                  std::int32_t *out_scores_cp) {
    gpu_eval::impl().infer(planes_batch, piece_counts, batch_size, out_scores_cp);
}
