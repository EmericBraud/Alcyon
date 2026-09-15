// Native MPSGraph implementation of v3's architecture (see
// training/cnn/legacy_model.py): stem conv+BN+ReLU (BN folded into the
// conv at export time, see export_weights_for_metal.py), 14 SE-gated
// residual blocks, 4 phase-bucket value heads, a PSQT skip head.
//
// Numerically verified against PyTorch via
// training/cnn/eval_compare/verify_export_numpy.py (max abs diff
// ~5e-4 on random inputs, float32-level noise) -- that script is the
// reference algorithm this file must keep matching bit-for-bit in
// structure. Runtime parity of *this* Objective-C++/MPSGraph
// implementation against that reference has NOT yet been verified
// end-to-end; do that (e.g. a small standalone test binary feeding the
// same random planes and comparing GpuBackend::infer_batch's output to
// verify_export_numpy.py's) before trusting this in a real search.
//
// All per-bucket heads (and the PSQT per-bucket scores) are computed
// for every sample regardless of its actual phase bucket, then
// selected via a one-hot dot product -- avoids any per-sample
// control-flow in the graph (MPSGraph has no cheap masked-gather-by-
// dynamic-index primitive worth using for only 4 buckets).
#include "gpu_backend.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cstdio>
#include <fstream>
#include <vector>

namespace gpu_eval {

namespace {

constexpr int kSEReduction = 4;

struct ConvWeights {
    std::vector<float> weight; // [out, in, 3, 3]
    std::vector<float> bias;   // [out]
};

struct LinearWeights {
    std::vector<float> weight_transposed; // [in, out] -- transposed from the exported [out, in]
    std::vector<float> bias;              // [out]
    int in_features = 0;
    int out_features = 0;
};

struct BlockWeights {
    ConvWeights conv1, conv2;
    LinearWeights se_fc1, se_fc2;
};

struct HeadWeights {
    LinearWeights fc1, fc2;
};

std::vector<float> read_floats(std::ifstream &f, std::size_t count) {
    std::vector<float> out(count);
    f.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return out;
}

LinearWeights read_linear(std::ifstream &f, int out_features, int in_features) {
    LinearWeights lw;
    lw.in_features = in_features;
    lw.out_features = out_features;
    std::vector<float> raw = read_floats(f, static_cast<std::size_t>(out_features) * in_features);
    lw.weight_transposed.resize(raw.size());
    for (int o = 0; o < out_features; ++o) {
        for (int i = 0; i < in_features; ++i) {
            lw.weight_transposed[static_cast<std::size_t>(i) * out_features + o] = raw[static_cast<std::size_t>(o) * in_features + i];
        }
    }
    lw.bias = read_floats(f, static_cast<std::size_t>(out_features));
    return lw;
}

} // namespace

class MetalGraphImpl {
public:
    bool load(const std::string &weights_path) {
        std::ifstream f(weights_path, std::ios::binary);
        if (!f) {
            std::fprintf(stderr, "gpu_eval: cannot open weights file %s\n", weights_path.c_str());
            return false;
        }

        std::int32_t header[4];
        f.read(reinterpret_cast<char *>(header), sizeof(header));
        channels_ = header[0];
        num_blocks_ = header[1];
        num_planes_ = header[2];
        num_buckets_ = header[3];

        stem_.weight = read_floats(f, static_cast<std::size_t>(channels_) * num_planes_ * 9);
        stem_.bias = read_floats(f, static_cast<std::size_t>(channels_));

        blocks_.resize(num_blocks_);
        for (auto &block : blocks_) {
            block.conv1.weight = read_floats(f, static_cast<std::size_t>(channels_) * channels_ * 9);
            block.conv1.bias = read_floats(f, static_cast<std::size_t>(channels_));
            block.conv2.weight = read_floats(f, static_cast<std::size_t>(channels_) * channels_ * 9);
            block.conv2.bias = read_floats(f, static_cast<std::size_t>(channels_));
            const int reduced = channels_ / kSEReduction;
            block.se_fc1 = read_linear(f, reduced, channels_);
            block.se_fc2 = read_linear(f, channels_, reduced);
        }

        if (num_buckets_ != 4) {
            // build_graph()'s head concatenation below is hand-unrolled for
            // exactly 4 buckets (v3's scheme) -- fail loudly rather than
            // silently mis-select heads for a differently-bucketed export.
            std::fprintf(stderr, "gpu_eval: unsupported num_phase_buckets=%d (only 4 supported)\n", num_buckets_);
            return false;
        }
        heads_.resize(num_buckets_);
        for (auto &head : heads_) {
            head.fc1 = read_linear(f, channels_, channels_ * 2);
            head.fc2 = read_linear(f, 1, channels_);
        }

        psqt_weight_ = read_floats(f, static_cast<std::size_t>(num_buckets_) * 12);

        if (!f) {
            std::fprintf(stderr, "gpu_eval: truncated/corrupt weights file %s\n", weights_path.c_str());
            return false;
        }

        return build_graph();
    }

    bool is_ready() const { return ready_; }
    int num_planes() const { return num_planes_; }

    void infer(const float *planes_batch, const int *piece_counts, int batch_size, std::int32_t *out_scores_cp) {
        @autoreleasepool {
            // An NSException here (shape mismatch, invalid MPSGraph feed,
            // ...) must NEVER escape this function: it would try to
            // propagate through the extern "C" boundary (gpu_backend.hpp)
            // into gpu_queue.cpp, which is compiled with -fno-exceptions --
            // undefined behavior, observed in practice as an immediate
            // std::terminate() that kills the whole engine process, not
            // just this background thread. Fail soft instead: log and
            // zero-fill scores.
            @try {
                infer_unsafe(planes_batch, piece_counts, batch_size, out_scores_cp);
            } @catch (NSException *exception) {
                std::fprintf(stderr, "gpu_eval: NSException during inference: %s -- %s\n",
                             [[exception name] UTF8String], [[exception reason] UTF8String]);
                for (int i = 0; i < batch_size; ++i) {
                    out_scores_cp[i] = 0;
                }
            }
        }
    }

private:
    void infer_unsafe(const float *planes_batch, const int *piece_counts, int batch_size, std::int32_t *out_scores_cp) {

            std::vector<int> bucket(batch_size);
            std::vector<float> one_hot(static_cast<std::size_t>(batch_size) * num_buckets_, 0.0f);
            for (int i = 0; i < batch_size; ++i) {
                int b = 0;
                static constexpr int kBoundaries[3] = {24, 16, 8};
                for (int boundary : kBoundaries) {
                    if (piece_counts[i] < boundary) {
                        ++b;
                    }
                }
                bucket[i] = b;
                one_hot[static_cast<std::size_t>(i) * num_buckets_ + b] = 1.0f;
            }

            // Build via MPSNDArray (writeBytes:strideBytes:) rather than
            // MPSGraphTensorData's raw NSData initializer -- the latter
            // hit an internal Metal-driver crash ("-[AGXG16GDevice
            // metalDevice]: unrecognized selector") on this hardware/OS
            // combination when exercised end-to-end; this is the more
            // standard/robust construction path.
            MPSNDArrayDescriptor *planes_desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                        shape:@[ @(batch_size), @(num_planes_), @8, @8 ]];
            MPSNDArray *planes_ndarray = [[MPSNDArray alloc] initWithDevice:mps_device_ descriptor:planes_desc];
            [planes_ndarray writeBytes:const_cast<float *>(planes_batch) strideBytes:nil];
            MPSGraphTensorData *planes_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:planes_ndarray];

            MPSNDArrayDescriptor *one_hot_desc = [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                                                         shape:@[ @(batch_size), @(num_buckets_) ]];
            MPSNDArray *one_hot_ndarray = [[MPSNDArray alloc] initWithDevice:mps_device_ descriptor:one_hot_desc];
            [one_hot_ndarray writeBytes:one_hot.data() strideBytes:nil];
            MPSGraphTensorData *one_hot_td = [[MPSGraphTensorData alloc] initWithMPSNDArray:one_hot_ndarray];

            NSDictionary *feeds = @{planes_placeholder_ : planes_td, one_hot_placeholder_ : one_hot_td};
            NSDictionary *results = [graph_ runWithFeeds:feeds targetTensors:@[ output_tensor_ ] targetOperations:nil];

            MPSGraphTensorData *out_td = results[output_tensor_];
            std::vector<float> out_scores(static_cast<std::size_t>(batch_size));
            [[out_td mpsndarray] readBytes:out_scores.data() strideBytes:nil];

            static constexpr float kScoreScale = 410.0f;
            for (int i = 0; i < batch_size; ++i) {
                out_scores_cp[i] = static_cast<std::int32_t>(out_scores[i] * kScoreScale);
            }
    }

private:
    MPSGraphTensor *make_constant(const std::vector<float> &data, NSArray<NSNumber *> *shape) {
        NSData *nsdata = [NSData dataWithBytes:data.data() length:sizeof(float) * data.size()];
        return [graph_ constantWithData:nsdata shape:shape dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor *conv3x3_bias(MPSGraphTensor *x, const ConvWeights &w, int out_channels, int in_channels) {
        MPSGraphTensor *weight = make_constant(w.weight, @[ @(out_channels), @(in_channels), @3, @3 ]);
        MPSGraphConvolution2DOpDescriptor *desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1
                          strideInY:1
                    dilationRateInX:1
                    dilationRateInY:1
                             groups:1
                       paddingLeft:1
                      paddingRight:1
                        paddingTop:1
                     paddingBottom:1
                      paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *conv = [graph_ convolution2DWithSourceTensor:x weightsTensor:weight descriptor:desc name:nil];
        MPSGraphTensor *bias = make_constant(w.bias, @[ @1, @(out_channels), @1, @1 ]);
        return [graph_ additionWithPrimaryTensor:conv secondaryTensor:bias name:nil];
    }

    MPSGraphTensor *linear(MPSGraphTensor *x, const LinearWeights &w) {
        MPSGraphTensor *weight = make_constant(w.weight_transposed, @[ @(w.in_features), @(w.out_features) ]);
        MPSGraphTensor *matmul = [graph_ matrixMultiplicationWithPrimaryTensor:x secondaryTensor:weight name:nil];
        MPSGraphTensor *bias = make_constant(w.bias, @[ @1, @(w.out_features) ]);
        return [graph_ additionWithPrimaryTensor:matmul secondaryTensor:bias name:nil];
    }

    MPSGraphTensor *global_avg_pool(MPSGraphTensor *x) {
        // x: [N, C, 8, 8] -> [N, C]
        MPSGraphTensor *pooled = [graph_ meanOfTensor:x axes:@[ @2, @3 ] name:nil];
        return [graph_ reshapeTensor:pooled withShape:@[ @-1, pooled.shape[1] ] name:nil];
    }

    bool build_graph() {
        graph_ = [[MPSGraph alloc] init];
        mps_device_ = MTLCreateSystemDefaultDevice();
        if (!mps_device_) {
            std::fprintf(stderr, "gpu_eval: no Metal device available\n");
            return false;
        }

        planes_placeholder_ = [graph_ placeholderWithShape:@[ @-1, @(num_planes_), @8, @8 ] dataType:MPSDataTypeFloat32 name:@"planes"];
        one_hot_placeholder_ = [graph_ placeholderWithShape:@[ @-1, @(num_buckets_) ] dataType:MPSDataTypeFloat32 name:@"bucket_one_hot"];

        MPSGraphTensor *x = [graph_ reLUWithTensor:conv3x3_bias(planes_placeholder_, stem_, channels_, num_planes_) name:nil];

        MPSGraphTensor *pooled_mid = nil;
        const int mid_point = num_blocks_ / 2;
        for (int i = 0; i < num_blocks_; ++i) {
            const BlockWeights &block = blocks_[i];
            MPSGraphTensor *residual = x;
            MPSGraphTensor *out = [graph_ reLUWithTensor:conv3x3_bias(x, block.conv1, channels_, channels_) name:nil];
            out = conv3x3_bias(out, block.conv2, channels_, channels_);

            MPSGraphTensor *pooled = global_avg_pool(out);
            MPSGraphTensor *gate = [graph_ sigmoidWithTensor:linear([graph_ reLUWithTensor:linear(pooled, block.se_fc1) name:nil], block.se_fc2) name:nil];
            MPSGraphTensor *gate_4d = [graph_ reshapeTensor:gate withShape:@[ @-1, @(channels_), @1, @1 ] name:nil];
            out = [graph_ multiplicationWithPrimaryTensor:out secondaryTensor:gate_4d name:nil];

            x = [graph_ reLUWithTensor:[graph_ additionWithPrimaryTensor:out secondaryTensor:residual name:nil] name:nil];

            if (i == mid_point - 1) {
                pooled_mid = global_avg_pool(x);
            }
        }
        MPSGraphTensor *pooled_final = global_avg_pool(x);
        MPSGraphTensor *head_input = [graph_ concatTensor:pooled_mid withTensor:pooled_final dimension:1 name:nil];

        std::vector<MPSGraphTensor *> head_outputs;
        head_outputs.reserve(static_cast<std::size_t>(num_buckets_));
        for (const HeadWeights &head : heads_) {
            MPSGraphTensor *h = [graph_ reLUWithTensor:linear(head_input, head.fc1) name:nil];
            MPSGraphTensor *out = linear(h, head.fc2); // [N, 1]
            head_outputs.push_back(out);
        }
        MPSGraphTensor *heads_stacked = [graph_ concatTensors:@[ head_outputs[0], head_outputs[1], head_outputs[2], head_outputs[3] ] dimension:1 name:nil]; // [N, num_buckets]
        MPSGraphTensor *trunk_logits = [graph_ reductionSumWithTensor:[graph_ multiplicationWithPrimaryTensor:heads_stacked secondaryTensor:one_hot_placeholder_ name:nil] axis:1 name:nil]; // [N, 1]

        // PSQT: 1x1 conv (12 -> num_buckets) over the piece planes only, summed spatially, then bucket-selected the same way.
        MPSGraphTensor *piece_planes = [graph_ sliceTensor:planes_placeholder_ dimension:1 start:0 length:12 name:nil];
        MPSGraphTensor *psqt_weight = make_constant(psqt_weight_, @[ @(num_buckets_), @12, @1, @1 ]);
        MPSGraphConvolution2DOpDescriptor *psqt_desc = [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:1
                          strideInY:1
                    dilationRateInX:1
                    dilationRateInY:1
                             groups:1
                       paddingLeft:0
                      paddingRight:0
                        paddingTop:0
                     paddingBottom:0
                      paddingStyle:MPSGraphPaddingStyleExplicit
                         dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                      weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *psqt_conv = [graph_ convolution2DWithSourceTensor:piece_planes weightsTensor:psqt_weight descriptor:psqt_desc name:nil]; // [N, num_buckets, 8, 8]
        MPSGraphTensor *psqt_per_bucket = [graph_ reshapeTensor:[graph_ meanOfTensor:psqt_conv axes:@[ @2, @3 ] name:nil]
                                                     withShape:@[ @-1, @(num_buckets_) ]
                                                          name:nil];
        // meanOfTensor gives the mean, but PSQT wants the SPATIAL SUM (see
        // legacy_model.py's PSQTHead: `.sum(dim=(2,3))`), so scale back up
        // by the plane area (8*8=64) to turn mean-of-64 back into sum-of-64.
        MPSGraphTensor *psqt_sum_per_bucket = [graph_ multiplicationWithPrimaryTensor:psqt_per_bucket
                                                                       secondaryTensor:[graph_ constantWithScalar:64.0 dataType:MPSDataTypeFloat32]
                                                                                  name:nil];
        MPSGraphTensor *psqt_logits = [graph_ reductionSumWithTensor:[graph_ multiplicationWithPrimaryTensor:psqt_sum_per_bucket secondaryTensor:one_hot_placeholder_ name:nil] axis:1 name:nil]; // [N, 1]

        output_tensor_ = [graph_ reshapeTensor:[graph_ additionWithPrimaryTensor:trunk_logits secondaryTensor:psqt_logits name:nil] withShape:@[ @-1 ] name:nil];

        ready_ = true;
        return true;
    }

    MPSGraph *graph_ = nil;
    id<MTLDevice> mps_device_ = nil;
    MPSGraphTensor *planes_placeholder_ = nil;
    MPSGraphTensor *one_hot_placeholder_ = nil;
    MPSGraphTensor *output_tensor_ = nil;

    int channels_ = 0, num_blocks_ = 0, num_planes_ = 0, num_buckets_ = 0;
    ConvWeights stem_;
    std::vector<BlockWeights> blocks_;
    std::vector<HeadWeights> heads_;
    std::vector<float> psqt_weight_;
    bool ready_ = false;
};

namespace {
MetalGraphImpl &impl() {
    static MetalGraphImpl instance;
    return instance;
}
} // namespace

} // namespace gpu_eval

// The extern "C" boundary (see gpu_backend.hpp) -- POD types only, no
// std::string/std::vector crossing into the rest of chess_core, which is
// compiled by a different C++ toolchain (this file needs Apple Clang;
// the rest of the project doesn't -- see that header's comment).
bool chess26_gpu_metal_load_weights(const char *weights_path) {
    return gpu_eval::impl().load(std::string(weights_path));
}

bool chess26_gpu_metal_is_ready() { return gpu_eval::impl().is_ready(); }

void chess26_gpu_metal_infer_batch(const float *planes_batch, const int *piece_counts, int batch_size,
                                      std::int32_t *out_scores_cp) {
    gpu_eval::impl().infer(planes_batch, piece_counts, batch_size, out_scores_cp);
}
