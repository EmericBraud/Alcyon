#pragma once

// NNUE v2 model: implements the "Full_Threats + HalfKAv2_hm^" combined feature
// set and the layer-stack architecture produced by nnue-pytorch commit
// 4289208fe20cc6ec8753e5ee14c2f210de783ff0 with default hyperparameters
// (L1=1024, L2=32, L3=32, 8 PSQT buckets, 8 layer-stack buckets). Confirmed
// against data/nnue/v2.nnue's header hash (see nnue_eval.hpp for the
// verification note).
//
// This differs from v1's layer-stack forward() (nnue_model.hpp) in several
// material ways (verified against model/modules/layer_stacks.py and
// model/quantize.py at the above commit):
//   - The L1 stacked-linear layer outputs exactly L2 columns (not L2+1): the
//     "skip" term is `l1_raw[L2-2] - l1_raw[L2-1]`, i.e. the last two of the
//     normal L2 outputs, not a dedicated extra column.
//   - L2's squared-CReLU output (both the squared and raw halves) is
//     concatenated with L1's squared-CReLU output to form the *output* layer's
//     input (2*L2 + 2*L3 = 128 wide), not just L2's output alone.
//   - Weight scales differ (L1 uses weight_scale_l1=128 => 2^7, not 2^6).
// Both networks share the same FT "pairwise square" (double_feature_transform)
// input stage and the same final PSQT-combination/output-descale arithmetic,
// which is why AccumulatorLayer/PsqtAccumulatorLayer/DenseLayer are reused
// as-is.

#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

#include "common/simd.hpp"
#include "core/piece/color.hpp"
#include "engine/eval/nnue/accumulator_layer.hpp"
#include "engine/eval/nnue/psqt_accumulator_layer.hpp"
#include "engine/eval/nnue/dense_layer.hpp"

template <int NFeatures, int NAccumulator = 1024, int NumPsqtBuckets = 8, int NumLsBuckets = 8, int L2 = 32, int L3 = 32>
class NnueModel
{
    static_assert(NAccumulator % 2 == 0, "NAccumulator must be even (split in half for the L0 pairwise square)");
    static_assert(L2 >= 2, "L1 stacked-linear output must have at least 2 columns for the skip term");

    static constexpr int WeightScaleBitsL1 = 7; // weight_scale_l1 = 128 = 2^7
    static constexpr int WeightScaleBitsL2 = 6; // weight_scale_l2 = 64 = 2^6
    // Final centipawn conversion: see get_result() below. Both networks share
    // weight_scale_out=16, so FinalScale = 2 * weight_scale_out = 32.
    static constexpr int FinalScale = 32;
    // Rescales the combined (output + skip) raw value, at scale
    // weight_scale_l1 * hidden_quantized_one = 128 * 128 = 16384, down to the
    // PSQT scale nnue2score * weight_scale_out = 600 * 16 = 9600.
    // 9600 / 16384 reduces exactly to 75 / 128.
    static constexpr int OutputRescaleNum = 75;
    static constexpr int OutputRescaleDen = 128;

public:
    using L1Layer = DenseLayer<NAccumulator, L2, WeightScaleBitsL1>;
    using L2Layer = DenseLayer<2 * L2, L3, WeightScaleBitsL2>;
    using OutputLayer = DenseLayer<2 * L2 + 2 * L3, 1>;

    struct LayerStackBucket
    {
        L1Layer l1;
        L2Layer l2;
        OutputLayer output;

        LayerStackBucket(L1Layer &&l1_, L2Layer &&l2_, OutputLayer &&output_)
            : l1(std::move(l1_)), l2(std::move(l2_)), output(std::move(output_))
        {
        }
    };

private:
    AccumulatorLayer<NFeatures, NAccumulator> accumulator;
    PsqtAccumulatorLayer<NFeatures, NumPsqtBuckets> psqt_accumulator;
    std::array<LayerStackBucket, NumLsBuckets> layer_stacks;

    static int bucket_for_piece_count(int piece_count)
    {
        constexpr int PiecesPerBucket = 32 / NumLsBuckets;
        int bucket = (piece_count - 1) / PiecesPerBucket;
        return std::clamp(bucket, 0, NumLsBuckets - 1);
    }

    // Same "pairwise square" FT activation as v1 (double_feature_transform):
    // split each perspective's clamped accumulator in half and multiply the
    // halves together elementwise.
    static void compute_l0(
        const std::array<std::int16_t, NAccumulator> &acc_us,
        const std::array<std::int16_t, NAccumulator> &acc_them,
        std::array<std::int8_t, NAccumulator> &l0)
    {
        constexpr int Half = NAccumulator / 2;

        // Matches nnue-pytorch's double_feature_transform + ComposedFeatureTransformer.forward:
        // accumulator halves are clamped to [0, ft_quantized_max=255] (not 127 -- the FT
        // accumulator is quantized at ft_quantized_one=256, twice the hidden-layer scale of
        // 128), multiplied, and the raw product is divided by inference_l0_division_factor=512
        // (== ft_quantized_one^2 / hidden_quantized_one, since l0_correction_factor == 1 here),
        // not by 128 (2^7). No extra *1:27 factor is applied.
        // int32 lane-matched to int16_v's width (native_simd<int32_t> may have a
        // narrower native width than native_simd<int16_t>, e.g. 4 vs 8 lanes on
        // 128-bit NEON/SSE), so the widening simd_cast below needs equal lane counts.
        using int32_wide_v = stdx::fixed_size_simd<std::int32_t, simd::SimdSize16>;
        using int8_wide_v = stdx::fixed_size_simd<std::int8_t, simd::SimdSize16>;

        auto pairwise_square_half = [](const std::int16_t *acc_ptr, std::int8_t *out)
        {
            std::size_t i = 0;
            for (; i + simd::SimdSize16 <= static_cast<std::size_t>(Half); i += simd::SimdSize16)
            {
                simd::int16_v acc_a_16, acc_b_16;

                acc_a_16.copy_from(acc_ptr + i, stdx::element_aligned);
                acc_b_16.copy_from(acc_ptr + i + Half, stdx::element_aligned);

                acc_a_16 = stdx::clamp(acc_a_16, simd::int16_v(0), simd::int16_v(255));
                acc_b_16 = stdx::clamp(acc_b_16, simd::int16_v(0), simd::int16_v(255));

                int32_wide_v acc_a_32 = stdx::simd_cast<int32_wide_v>(acc_a_16);
                int32_wide_v acc_b_32 = stdx::simd_cast<int32_wide_v>(acc_b_16);

                int32_wide_v prod = (acc_a_32 * acc_b_32) >> 9;
                prod = stdx::clamp(prod, int32_wide_v(0), int32_wide_v(127));

                const int8_wide_v prod_8 = stdx::static_simd_cast<int8_wide_v>(prod);
                prod_8.copy_to(out + i, stdx::element_aligned);
            }
            for (; i < static_cast<std::size_t>(Half); ++i)
            {
                std::int32_t a = std::clamp<std::int32_t>(acc_ptr[i], 0, 255);
                std::int32_t b = std::clamp<std::int32_t>(acc_ptr[i + Half], 0, 255);
                std::int32_t prod = (a * b) / 512;
                out[i] = static_cast<std::int8_t>(std::clamp(prod, 0, 127));
            }
        };

        pairwise_square_half(acc_us.data(), l0.data());
        pairwise_square_half(acc_them.data(), l0.data() + Half);
    }

    // Squared-CReLU: given raw (unshifted) dense-layer output at scale
    // 2^WeightScaleBits * hidden_quantized_one(128), produces the
    // [squared-half | raw-half] activation pair used both as the next layer's
    // input and (reused) as part of the output layer's wider input.
    template <int N, int WeightScaleBits>
    static void squared_crelu(const std::array<std::int32_t, N> &raw, std::array<std::int8_t, 2 * N> &out)
    {
        for (int i = 0; i < N; ++i)
        {
            const std::int32_t x = raw[i] >> WeightScaleBits;
            const std::int32_t sqr = (x * x) >> 7;
            out[i] = static_cast<std::int8_t>(std::clamp(sqr, 0, 127));
            out[N + i] = static_cast<std::int8_t>(std::clamp(x, 0, 127));
        }
    }

public:
    template <typename AccBiases, typename AccWeights, typename PsqtWeights, typename LayerStacksArr>
    NnueModel(AccBiases &&acc_biases, AccWeights &&acc_weights, PsqtWeights &&psqt_weights, LayerStacksArr &&layer_stacks_)
        : accumulator(std::forward<AccBiases>(acc_biases), std::forward<AccWeights>(acc_weights)),
          psqt_accumulator(std::forward<PsqtWeights>(psqt_weights)),
          layer_stacks(std::forward<LayerStacksArr>(layer_stacks_))
    {
    }

    void reset()
    {
        accumulator.reset();
        psqt_accumulator.reset();
    }

    template <Color perspective>
    void reset_perspective()
    {
        accumulator.template reset_perspective<perspective>();
        psqt_accumulator.template reset_perspective<perspective>();
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        accumulator.template update_feature<activate, perspective>(feature);
        psqt_accumulator.template update_feature<activate, perspective>(feature);
    }

    // See AccumulatorLayer::prefetch -- call several iterations ahead of the
    // matching update_feature() for `feature` in a caller's loop.
    void prefetch_feature(int feature) const
    {
        accumulator.prefetch(feature);
        psqt_accumulator.prefetch(feature);
    }

    // piece_count: total number of pieces on the board (both colors, kings
    // included) -- selects both the PSQT bucket and the layer-stack bucket.
    template <Color perspective>
    std::int32_t get_result(int piece_count) const
    {
        constexpr Color us = perspective;
        constexpr Color them = !perspective;

        const auto &acc_us = accumulator.template get_accumulator<us>();
        const auto &acc_them = accumulator.template get_accumulator<them>();

        std::array<std::int8_t, NAccumulator> l0;
        compute_l0(acc_us, acc_them, l0);

        const int bucket = bucket_for_piece_count(piece_count);
        const auto &ls = layer_stacks[bucket];

        const auto raw_l1 = ls.l1.get_raw(l0);
        const std::int32_t skip_raw = raw_l1[L2 - 2] - raw_l1[L2 - 1];

        std::array<std::int8_t, 2 * L2> l1_out;
        squared_crelu<L2, WeightScaleBitsL1>(raw_l1, l1_out);

        const auto raw_l2 = ls.l2.get_raw(l1_out);
        std::array<std::int8_t, 2 * L3> l2_out;
        squared_crelu<L3, WeightScaleBitsL2>(raw_l2, l2_out);

        const std::int32_t output_raw = ls.output.get_result_split(l1_out, l2_out);

        const std::int64_t combined_raw = static_cast<std::int64_t>(output_raw) + static_cast<std::int64_t>(skip_raw);
        const std::int64_t layerstack_final =
            (combined_raw * OutputRescaleNum) / OutputRescaleDen; // truncating division, matches quantize.py's trunc mode

        const auto &psqt_us = psqt_accumulator.template get_accumulator<us>();
        const auto &psqt_them = psqt_accumulator.template get_accumulator<them>();
        const std::int64_t psqt_diff = static_cast<std::int64_t>(psqt_us[bucket]) - static_cast<std::int64_t>(psqt_them[bucket]);

        const std::int64_t combined = 2 * layerstack_final + psqt_diff;
        return static_cast<std::int32_t>(combined / FinalScale);
    }

#ifdef CHESS26_UNIT_TESTING
    const auto &get_accumulator() const
    {
        return accumulator;
    }
#endif
};
