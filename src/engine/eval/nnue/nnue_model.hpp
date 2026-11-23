#pragma once
#include <vector>

// NNUE v2 model: implements the "Full_Threats + HalfKAv2_hm^" combined feature
// set and the layer-stack architecture produced by nnue-pytorch commit
// 4289208fe20cc6ec8753e5ee14c2f210de783ff0 with default hyperparameters
// (L1=1024, L2=32, L3=32, 8 PSQT buckets, 8 layer-stack buckets). Confirmed
// against data/nnue/v3.nnue's header hash (see nnue_eval.hpp for the
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
#include <memory>
#include <utility>

#include "common/constants.hpp"
#include "common/simd.hpp"
#include "core/piece/color.hpp"
#include "engine/eval/nnue/accumulator_layer.hpp"
#include "engine/eval/nnue/psqt_accumulator_layer.hpp"
#include "engine/eval/nnue/dense_layer.hpp"

template <int NFeatures, int NThreatFeatures, int NAccumulator = 1024, int NumPsqtBuckets = 8, int NumLsBuckets = 8, int L2 = 32, int L3 = 32>
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
    AccumulatorLayer<NFeatures, NAccumulator, NThreatFeatures> accumulator;
    PsqtAccumulatorLayer<NFeatures, NumPsqtBuckets> psqt_accumulator;
    std::array<LayerStackBucket, NumLsBuckets> layer_stacks;

    // Snapshot stacks backing push_acc_state()/push_psqt_state() (see
    // below): each entry is a full copy of both perspectives' state (~4KB
    // for the L1 accumulator, 64B for the PSQT buckets), allocated once up
    // front (constants::MaxHistorySize deep -- the same bound Board's own
    // move-history array uses, so nesting can never exceed it) rather than
    // per-call, keeping push/pop allocation-free on the hot path. The two
    // stacks have *independent* depths: the PSQT half materializes on cheap
    // pruning evals (NnueEval::materialize_psqt) and the L1 half only on
    // full evals, so at any moment each half may have caught up to a
    // different ply.
    using AccSnapshot = typename AccumulatorLayer<NFeatures, NAccumulator, NThreatFeatures>::AccTable;
    using PsqtSnapshot = typename PsqtAccumulatorLayer<NFeatures, NumPsqtBuckets>::AccTable;
    std::unique_ptr<AccSnapshot[]> acc_snapshots = std::make_unique<AccSnapshot[]>(constants::MaxHistorySize);
    std::unique_ptr<PsqtSnapshot[]> psqt_snapshots = std::make_unique<PsqtSnapshot[]>(constants::MaxHistorySize);
    int acc_snapshot_depth = 0;
    int psqt_snapshot_depth = 0;

    // Deep-copies only the live snapshot prefixes -- the rest is unwritten
    // scratch space, not meaningful state to preserve.
    void copy_snapshots_from(const NnueModel &other)
    {
        acc_snapshot_depth = other.acc_snapshot_depth;
        psqt_snapshot_depth = other.psqt_snapshot_depth;
        std::copy_n(other.acc_snapshots.get(), acc_snapshot_depth, acc_snapshots.get());
        std::copy_n(other.psqt_snapshots.get(), psqt_snapshot_depth, psqt_snapshots.get());
    }

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
        std::array<std::uint8_t, NAccumulator> &l0)
    {
        constexpr int Half = NAccumulator / 2;

        // Matches nnue-pytorch's double_feature_transform + ComposedFeatureTransformer.forward:
        // accumulator halves are clamped to [0, ft_quantized_max=255] (not 127 -- the FT
        // accumulator is quantized at ft_quantized_one=256, twice the hidden-layer scale of
        // 128), multiplied, and the raw product is divided by inference_l0_division_factor=512
        // (== ft_quantized_one^2 / hidden_quantized_one, since l0_correction_factor == 1 here),
        // not by 128 (2^7). No extra *1:27 factor is applied.
        // The whole pipeline stays in 16-bit lanes: both factors are in
        // [0, 255] after the clamp, so their product fits a uint16
        // (255*255 = 65025 <= 65535) and (a*b) >> 9 <= 127 by construction
        // -- no widening to int32 and no output clamp needed.
        using uint16_v = stdx::rebind_simd_t<std::uint16_t, simd::int16_v>;
        using uint8_wide_v = stdx::fixed_size_simd<std::uint8_t, simd::SimdSize16>;

        auto pairwise_square_half = [](const std::int16_t *acc_ptr, std::uint8_t *out)
        {
            std::size_t i = 0;
            for (; i + simd::SimdSize16 <= static_cast<std::size_t>(Half); i += simd::SimdSize16)
            {
                simd::int16_v acc_a_16, acc_b_16;

                acc_a_16.copy_from(acc_ptr + i, stdx::element_aligned);
                acc_b_16.copy_from(acc_ptr + i + Half, stdx::element_aligned);

                acc_a_16 = stdx::clamp(acc_a_16, simd::int16_v(0), simd::int16_v(255));
                acc_b_16 = stdx::clamp(acc_b_16, simd::int16_v(0), simd::int16_v(255));

                const uint16_v acc_a_u = stdx::static_simd_cast<uint16_v>(acc_a_16);
                const uint16_v acc_b_u = stdx::static_simd_cast<uint16_v>(acc_b_16);

                const uint16_v prod = (acc_a_u * acc_b_u) >> 9;

                const uint8_wide_v prod_8 = stdx::static_simd_cast<uint8_wide_v>(prod);
                prod_8.copy_to(out + i, stdx::element_aligned);
            }
            for (; i < static_cast<std::size_t>(Half); ++i)
            {
                std::int32_t a = std::clamp<std::int32_t>(acc_ptr[i], 0, 255);
                std::int32_t b = std::clamp<std::int32_t>(acc_ptr[i + Half], 0, 255);
                std::int32_t prod = (a * b) / 512;
                out[i] = static_cast<std::uint8_t>(std::clamp(prod, 0, 127));
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
    static void squared_crelu(const std::array<std::int32_t, N> &raw, std::array<std::uint8_t, 2 * N> &out)
    {
        for (int i = 0; i < N; ++i)
        {
            const std::int32_t x = raw[i] >> WeightScaleBits;
            const std::int32_t sqr = (x * x) >> 7;
            out[i] = static_cast<std::uint8_t>(std::clamp(sqr, 0, 127));
            out[N + i] = static_cast<std::uint8_t>(std::clamp(x, 0, 127));
        }
    }

public:
    template <typename AccBiases, typename AccWeightsInt8, typename AccWeightsInt16, typename PsqtWeights, typename LayerStacksArr>
    NnueModel(AccBiases &&acc_biases, AccWeightsInt8 &&acc_weights_int8, AccWeightsInt16 &&acc_weights_int16, PsqtWeights &&psqt_weights, LayerStacksArr &&layer_stacks_)
        : accumulator(std::forward<AccBiases>(acc_biases), std::forward<AccWeightsInt8>(acc_weights_int8), std::forward<AccWeightsInt16>(acc_weights_int16)),
          psqt_accumulator(std::forward<PsqtWeights>(psqt_weights)),
          layer_stacks(std::forward<LayerStacksArr>(layer_stacks_))
    {
    }

    // Declared explicitly: acc_snapshots/psqt_snapshots are std::unique_ptr<T[]>,
    // which isn't copyable, so the implicit copy-ctor/assignment would
    // otherwise be deleted (see AccumulatorLayer's analogous note). Only the
    // live [0, snapshot_depth) prefix is copied (copy_snapshots_from) -- a
    // copy only ever happens at thread/VBoard setup (see VBoard's copy
    // paths), never on the play()/unplay() hot path, so this cost is
    // amortized to nothing.
    NnueModel(const NnueModel &other)
        : accumulator(other.accumulator),
          psqt_accumulator(other.psqt_accumulator),
          layer_stacks(other.layer_stacks)
    {
        copy_snapshots_from(other);
    }

    NnueModel &operator=(const NnueModel &other)
    {
        if (this != &other)
        {
            accumulator = other.accumulator;
            psqt_accumulator = other.psqt_accumulator;
            layer_stacks = other.layer_stacks;
            copy_snapshots_from(other);
        }
        return *this;
    }

    NnueModel(NnueModel &&) noexcept = default;
    NnueModel &operator=(NnueModel &&) noexcept = default;

    void reset()
    {
        accumulator.reset();
        psqt_accumulator.reset();
    }

    // Combined both-halves variants: used by the full-rebuild path
    // (NnueEval::initialize_perspective), where the L1 accumulator and the
    // PSQT buckets are always rebuilt together.
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

    // Single-half variants: the lazy materialization paths catch each half
    // up independently (PSQT on cheap pruning evals, L1 on full evals), so
    // each applies its own half of a buffered diff -- see NnueEval::
    // materialize / materialize_psqt.
    template <Color perspective>
    void reset_acc_perspective()
    {
        accumulator.template reset_perspective<perspective>();
    }

    template <Color perspective>
    void reset_psqt_perspective()
    {
        psqt_accumulator.template reset_perspective<perspective>();
    }

    template <bool activate, Color perspective>
    void update_acc_feature(int feature)
    {
        accumulator.template update_feature<activate, perspective>(feature);
    }

    template <bool activate, Color perspective, int K>
    void update_acc_feature_narrow(int feature)
    {
        accumulator.template update_feature_narrow<activate, perspective, K>(feature);
    }

    template <bool activate, Color perspective>
    void update_psqt_feature(int feature)
    {
        psqt_accumulator.template update_feature<activate, perspective>(feature);
    }

    void prefetch_acc_feature(int feature) const
    {
        accumulator.prefetch(feature);
    }

    void prefetch_psqt_feature(int feature) const
    {
        psqt_accumulator.prefetch(feature);
    }

    // Saves one half's state (both perspectives) onto its stack, restored
    // by the matching pop -- a plain memcpy (~4KB for the L1 accumulator,
    // 64B for PSQT), versus replaying collect+diff to undo a move's
    // incremental update. Callers must pair every push with exactly one
    // pop, in LIFO order per stack (mirrors Board's own play/unplay
    // nesting) -- see NnueEval::materialize/materialize_psqt/unplay_pop.
    void push_acc_state()
    {
        acc_snapshots[acc_snapshot_depth] = accumulator.raw_state();
        ++acc_snapshot_depth;
    }

    void pop_acc_state()
    {
        --acc_snapshot_depth;
        accumulator.restore_raw_state(acc_snapshots[acc_snapshot_depth]);
    }

    void push_psqt_state()
    {
        psqt_snapshots[psqt_snapshot_depth] = psqt_accumulator.raw_state();
        ++psqt_snapshot_depth;
    }

    void pop_psqt_state()
    {
        --psqt_snapshot_depth;
        psqt_accumulator.restore_raw_state(psqt_snapshots[psqt_snapshot_depth]);
    }

    // Drops every outstanding snapshot without restoring anything -- for
    // callers that are about to rebuild the accumulator from scratch (see
    // NnueEval::initialize) and therefore invalidate both stacks.
    void reset_snapshot_stack()
    {
        acc_snapshot_depth = 0;
        psqt_snapshot_depth = 0;
    }

    // Entree de la pile de couches (l0) du point de vue de `perspective`,
    // exposee pour le dump de docs/learned-pruning.md.
    template <Color perspective>
    void get_l0(std::array<std::uint8_t, NAccumulator> &l0) const
    {
        compute_l0(accumulator.template get_accumulator<perspective>(),
                   accumulator.template get_accumulator<!perspective>(), l0);
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

        std::array<std::uint8_t, NAccumulator> l0;
        compute_l0(acc_us, acc_them, l0);

        const int bucket = bucket_for_piece_count(piece_count);
        const auto &ls = layer_stacks[bucket];

        const auto raw_l1 = ls.l1.get_raw(l0);
        const std::int32_t skip_raw = raw_l1[L2 - 2] - raw_l1[L2 - 1];

        std::array<std::uint8_t, 2 * L2> l1_out;
        squared_crelu<L2, WeightScaleBitsL1>(raw_l1, l1_out);

        const auto raw_l2 = ls.l2.get_raw(l1_out);
        std::array<std::uint8_t, 2 * L3> l2_out;
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

    // Palier reduit (eval par paliers) : petite tete flottante distillee de
    // la tete complete (tools/gpu/tiers_export.py), qui ne lit que les
    // neurones [0, K) des deux perspectives d'un reseau permute par
    // importance. Les accumulateurs n'ont besoin d'etre a jour que sur
    // [0, K) et [Half, Half + K). Meme unite que get_result (centipions).
    template <Color perspective, class Head>
    std::int32_t get_narrow_result(int piece_count) const
    {
        constexpr int K = Head::kK;
        constexpr int Half = NAccumulator / 2;
        constexpr int In = 2 * K;
        const int bucket = bucket_for_piece_count(piece_count);
        const auto &acc_us = accumulator.template get_accumulator<perspective>();
        const auto &acc_them = accumulator.template get_accumulator<!perspective>();

        // Premiere couche en accumulation par colonnes (axpy) : vectorisable
        // sans reassocier de somme, et les entrees nulles de l0 (nombreuses :
        // ReLU puis produit) sont sautees. Poids transposes une fois [In][L2].
        static const auto l1t = []
        {
            auto t = std::make_unique<float[]>(std::size_t(Head::kBuckets) * In * Head::kL2);
            for (int b = 0; b < Head::kBuckets; ++b)
                for (int o = 0; o < Head::kL2; ++o)
                    for (int i = 0; i < In; ++i)
                        t[(std::size_t(b) * In + i) * Head::kL2 + o] = Head::kL1W[(b * Head::kL2 + o) * In + i];
            return t;
        }();
        float l1c[Head::kL2];
        std::copy_n(Head::kL1B + bucket * Head::kL2, Head::kL2, l1c);
        auto add_input = [&](int i, int v)
        {
            if (v == 0)
                return;
            const float xi = float(v) * (1.0f / 128.0f);
            const float *col = l1t.get() + (std::size_t(bucket) * In + i) * Head::kL2;
            for (int o = 0; o < Head::kL2; ++o)
                l1c[o] += col[o] * xi;
        };
        for (int j = 0; j < K; ++j)
        {
            const int a = std::clamp<int>(acc_us[j], 0, 255), b = std::clamp<int>(acc_us[j + Half], 0, 255);
            const int c = std::clamp<int>(acc_them[j], 0, 255), d = std::clamp<int>(acc_them[j + Half], 0, 255);
            add_input(j, (a * b) >> 9);
            add_input(K + j, (c * d) >> 9);
        }
        const float skip = l1c[Head::kL2 - 2] - l1c[Head::kL2 - 1];
        float l1x[2 * Head::kL2];
        for (int o = 0; o < Head::kL2; ++o)
        {
            l1x[o] = std::clamp(l1c[o] * l1c[o], 0.0f, Head::kMaxAct);
            l1x[Head::kL2 + o] = std::clamp(l1c[o], 0.0f, Head::kMaxAct);
        }
        // Deuxieme couche et sortie aussi en accumulation par colonnes (poids
        // transposes une fois) : pas de reduction flottante a reassocier.
        static const auto l2t = []
        {
            auto t = std::make_unique<float[]>(std::size_t(Head::kBuckets) * 2 * Head::kL2 * Head::kL3);
            for (int b = 0; b < Head::kBuckets; ++b)
                for (int o = 0; o < Head::kL3; ++o)
                    for (int i = 0; i < 2 * Head::kL2; ++i)
                        t[(std::size_t(b) * 2 * Head::kL2 + i) * Head::kL3 + o] = Head::kL2W[(b * Head::kL3 + o) * 2 * Head::kL2 + i];
            return t;
        }();
        float l2c[Head::kL3];
        std::copy_n(Head::kL2B + bucket * Head::kL3, Head::kL3, l2c);
        for (int i = 0; i < 2 * Head::kL2; ++i)
        {
            const float xi = l1x[i];
            if (xi == 0.0f)
                continue;
            const float *col = l2t.get() + (std::size_t(bucket) * 2 * Head::kL2 + i) * Head::kL3;
            for (int o = 0; o < Head::kL3; ++o)
                l2c[o] += col[o] * xi;
        }
        float l2x[2 * Head::kL3];
        for (int o = 0; o < Head::kL3; ++o)
        {
            l2x[o] = std::clamp(l2c[o] * l2c[o], 0.0f, Head::kMaxAct);
            l2x[Head::kL3 + o] = std::clamp(l2c[o], 0.0f, Head::kMaxAct);
        }
        // Sortie : produit scalaire en 8 sommes partielles independantes.
        const float *ow = Head::kOW + bucket * (2 * Head::kL2 + 2 * Head::kL3);
        float part[8] = {};
        for (int i = 0; i < 2 * Head::kL2; ++i)
            part[i & 7] += ow[i] * l1x[i];
        for (int i = 0; i < 2 * Head::kL3; ++i)
            part[i & 7] += ow[2 * Head::kL2 + i] * l2x[i];
        float out = Head::kOB[bucket] + skip;
        for (float p : part)
            out += p;

        const auto &psqt_us = psqt_accumulator.template get_accumulator<perspective>();
        const auto &psqt_them = psqt_accumulator.template get_accumulator<!perspective>();
        const std::int64_t psqt_diff = static_cast<std::int64_t>(psqt_us[bucket]) - static_cast<std::int64_t>(psqt_them[bucket]);
        return static_cast<std::int32_t>(psqt_diff / FinalScale) + static_cast<std::int32_t>(out * Head::kNnue2Score);
    }

    // Palier reduit quantifie : petite tete au format de la grande (int8 /
    // int32, memes echelles, quantize_fc_layer de nnue-pytorch), sur les K
    // premiers neurones des deux perspectives. Reutilise les couches denses
    // vectorisees et la meme chaine de sortie que get_result.
    template <Color perspective, class Q>
    std::int32_t get_narrow_result_q(int piece_count) const
    {
        constexpr int K = Q::kK;
        constexpr int Half = NAccumulator / 2;
        using SL1 = DenseLayer<2 * K, L2, WeightScaleBitsL1>;
        struct SmallStack
        {
            SL1 l1;
            L2Layer l2;
            OutputLayer output;
        };
        static const auto stacks = []
        {
            std::vector<SmallStack> v;
            v.reserve(NumLsBuckets);
            for (int b = 0; b < NumLsBuckets; ++b)
            {
                AlignedArray<std::array<std::int8_t, 2 * K>, L2> w1;
                AlignedArray<std::int32_t, L2> b1;
                for (int o = 0; o < L2; ++o)
                {
                    b1[o] = Q::kL1B[b * L2 + o];
                    for (int i = 0; i < 2 * K; ++i)
                        w1[o][i] = Q::kL1W[(b * L2 + o) * 2 * K + i];
                }
                AlignedArray<std::array<std::int8_t, 2 * L2>, L3> w2;
                AlignedArray<std::int32_t, L3> b2;
                for (int o = 0; o < L3; ++o)
                {
                    b2[o] = Q::kL2B[b * L3 + o];
                    for (int i = 0; i < 2 * L2; ++i)
                        w2[o][i] = Q::kL2W[(b * L3 + o) * 2 * L2 + i];
                }
                AlignedArray<std::array<std::int8_t, 2 * L2 + 2 * L3>, 1> wo;
                AlignedArray<std::int32_t, 1> bo;
                bo[0] = Q::kOB[b];
                for (int i = 0; i < 2 * L2 + 2 * L3; ++i)
                    wo[0][i] = Q::kOW[b * (2 * L2 + 2 * L3) + i];
                v.push_back(SmallStack{SL1(std::move(w1), std::move(b1)), L2Layer(std::move(w2), std::move(b2)),
                                       OutputLayer(std::move(wo), std::move(bo))});
            }
            return v;
        }();

        const int bucket = bucket_for_piece_count(piece_count);
        const auto &acc_us = accumulator.template get_accumulator<perspective>();
        const auto &acc_them = accumulator.template get_accumulator<!perspective>();
        std::array<std::uint8_t, 2 * K> l0;
        for (int j = 0; j < K; ++j)
        {
            const int a = std::clamp<int>(acc_us[j], 0, 255), b = std::clamp<int>(acc_us[j + Half], 0, 255);
            const int c = std::clamp<int>(acc_them[j], 0, 255), d = std::clamp<int>(acc_them[j + Half], 0, 255);
            l0[j] = std::uint8_t((a * b) >> 9);
            l0[K + j] = std::uint8_t((c * d) >> 9);
        }
        const auto &ls = stacks[bucket];
        const auto raw_l1 = ls.l1.get_raw(l0);
        const std::int32_t skip_raw = raw_l1[L2 - 2] - raw_l1[L2 - 1];
        std::array<std::uint8_t, 2 * L2> l1_out;
        squared_crelu<L2, WeightScaleBitsL1>(raw_l1, l1_out);
        const auto raw_l2 = ls.l2.get_raw(l1_out);
        std::array<std::uint8_t, 2 * L3> l2_out;
        squared_crelu<L3, WeightScaleBitsL2>(raw_l2, l2_out);
        const std::int32_t output_raw = ls.output.get_result_split(l1_out, l2_out);
        const std::int64_t combined_raw = static_cast<std::int64_t>(output_raw) + static_cast<std::int64_t>(skip_raw);
        const std::int64_t layerstack_final = (combined_raw * OutputRescaleNum) / OutputRescaleDen;
        const auto &psqt_us = psqt_accumulator.template get_accumulator<perspective>();
        const auto &psqt_them = psqt_accumulator.template get_accumulator<!perspective>();
        const std::int64_t psqt_diff = static_cast<std::int64_t>(psqt_us[bucket]) - static_cast<std::int64_t>(psqt_them[bucket]);
        return static_cast<std::int32_t>((2 * layerstack_final + psqt_diff) / FinalScale);
    }

    // PSQT-only output: the material/PSQT head of the network, skipping the
    // whole layer stack (compute_l0 + 3 dense layers). Same units and same
    // bucket selection as get_result() -- get_result computes
    // (2*layerstack + psqt_diff) / FinalScale, so the psqt_diff / FinalScale
    // term alone IS the network's trained material/PSQT estimate in
    // centipawns. Used as the cheap pruning eval (see Eval::
    // lazy_eval_relative) in place of the HCE EvalState estimate.
    template <Color perspective>
    std::int32_t get_psqt_result(int piece_count) const
    {
        constexpr Color us = perspective;
        constexpr Color them = !perspective;
        const int bucket = bucket_for_piece_count(piece_count);
        const auto &psqt_us = psqt_accumulator.template get_accumulator<us>();
        const auto &psqt_them = psqt_accumulator.template get_accumulator<them>();
        const std::int64_t psqt_diff = static_cast<std::int64_t>(psqt_us[bucket]) - static_cast<std::int64_t>(psqt_them[bucket]);
        return static_cast<std::int32_t>(psqt_diff / FinalScale);
    }

#ifdef ALCYON_UNIT_TESTING
    const auto &get_accumulator() const
    {
        return accumulator;
    }
#endif
};
