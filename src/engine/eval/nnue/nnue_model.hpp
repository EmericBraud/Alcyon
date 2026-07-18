#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <algorithm>

#include "common/constants.hpp"
#include "common/mask.hpp"
#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "engine/eval/nnue/features_encoder.hpp"
#include "engine/eval/nnue/accumulator_layer.hpp"
#include "engine/eval/nnue/psqt_accumulator_layer.hpp"
#include "engine/eval/nnue/dense_layer.hpp"

// Implements the "squared-CReLU with skip connection" layer-stack architecture
// used by nnue-pytorch (feature transformer -> pairwise-square L0 -> 8
// material-bucketed FC stacks with a linear skip path -> PSQT combination).
// Reference: nnue-pytorch commit 00bdf75 (model/model.py, model/utils/serialize.py),
// which matches this project's .nnue format (version 0x7AF32F20).
template <int NFeatures, int NAccumulator = 256, int NumPsqtBuckets = 8, int NumLsBuckets = 8, int L2 = 32, int L3 = 32>
class NnueModel
{
    static_assert(NAccumulator % 2 == 0, "NAccumulator must be even (split in half for the L0 pairwise square)");

    // Quantization scales, matching nnue-pytorch's NNUEModel (nnue2score=600,
    // weight_scale_hidden=64, weight_scale_out=16, quantized_one=127).
    static constexpr int WeightScaleBits = 6; // 2^6 = weight_scale_hidden
    // Rescales the L1 skip output (hidden bias scale = 64*127 = 8128) to the
    // output-layer bias scale (weight_scale_out*nnue2score = 16*600 = 9600).
    // 9600/8128 reduces exactly to 150/127.
    static constexpr int SkipRescaleNum = 150;
    static constexpr int SkipRescaleDen = 127;
    // Final centipawn conversion combines the /16 output scale with the 0.5
    // perspective-averaging factor on the PSQT term: see get_result() below.
    static constexpr int FinalScale = 32;

public:
    using L1Layer = DenseLayer<NAccumulator, L2 + 1, WeightScaleBits>;
    using L2Layer = DenseLayer<2 * L2, L3, WeightScaleBits>;
    using OutputLayer = DenseLayer<L3, 1>;

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

    // Feeds each perspective's clamped accumulator through the "pairwise
    // square" activation: split the L1-wide accumulator in half and multiply
    // the two halves together (elementwise), separately for us/them, then
    // concatenate. This replaces the classic single ClippedReLU input layer.
    static void compute_l0(
        const std::array<std::int16_t, NAccumulator> &acc_us,
        const std::array<std::int16_t, NAccumulator> &acc_them,
        std::array<std::int8_t, NAccumulator> &l0)
    {
        constexpr int Half = NAccumulator / 2;

        auto pairwise_square_half = [](const std::array<std::int16_t, NAccumulator> &acc, std::int8_t *out)
        {
            for (int i = 0; i < Half; ++i)
            {
                std::int32_t a = std::clamp<std::int32_t>(acc[i], 0, 127);
                std::int32_t b = std::clamp<std::int32_t>(acc[i + Half], 0, 127);
                std::int32_t prod = (a * b * 127) >> 7;
                out[i] = static_cast<std::int8_t>(std::clamp(prod, 0, 127));
            }
        };

        pairwise_square_half(acc_us, l0.data());
        pairwise_square_half(acc_them, l0.data() + Half);
    }

public:
    template <typename AccBiases, typename AccWeights, typename PsqtWeights, typename LayerStacksArr>
    NnueModel(AccBiases &&acc_biases, AccWeights &&acc_weights, PsqtWeights &&psqt_weights, LayerStacksArr &&layer_stacks_)
        : accumulator(std::forward<AccBiases>(acc_biases), std::forward<AccWeights>(acc_weights)),
          psqt_accumulator(std::forward<PsqtWeights>(psqt_weights)),
          layer_stacks(std::forward<LayerStacksArr>(layer_stacks_))
    {
    }

    void initialize(const std::array<U64, constants::NumPieceVariants> &occupancies)
    {
        this->accumulator.reset();
        this->psqt_accumulator.reset();

        int white_king_sq, black_king_sq;
        {
            U64 white_king_occ = occupancies[KING];
            assert(white_king_occ);
            white_king_sq = cpu::pop_lsb(white_king_occ);
        }
        {
            U64 black_king_occ = occupancies[KING + constants::PieceTypeCount];
            assert(black_king_occ);
            black_king_sq = cpu::pop_lsb(black_king_occ);
        }
        for (int piece_color = WHITE; piece_color <= BLACK; ++piece_color)
        {
            for (int piece_type = PAWN; piece_type <= KING; ++piece_type)
            {
                U64 occupancy = occupancies[piece_type + piece_color * constants::PieceTypeCount];
                while (occupancy != 0ULL)
                {
                    int sq = cpu::pop_lsb(occupancy);
                    int feature = feature_encoder::get_feature_index<WHITE>(white_king_sq, static_cast<Color>(piece_color), static_cast<Piece>(piece_type), sq);
                    accumulator.template update_feature<true, WHITE>(feature);
                    psqt_accumulator.template update_feature<true, WHITE>(feature);
                }
            }
        }
        for (int piece_color = WHITE; piece_color <= BLACK; ++piece_color)
        {
            for (int piece_type = PAWN; piece_type <= KING; ++piece_type)
            {
                U64 occupancy = occupancies[piece_type + piece_color * constants::PieceTypeCount];
                while (occupancy != 0ULL)
                {
                    int sq = cpu::pop_lsb(occupancy);
                    int feature = feature_encoder::get_feature_index<BLACK>(black_king_sq, static_cast<Color>(piece_color), static_cast<Piece>(piece_type), sq);
                    accumulator.template update_feature<true, BLACK>(feature);
                    psqt_accumulator.template update_feature<true, BLACK>(feature);
                }
            }
        }
    }

    // piece_count: total number of pieces on the board (both colors, kings
    // included) — selects both the PSQT bucket and the layer-stack bucket
    // (they always share the same index: (piece_count - 1) / (32 / buckets)).
    template <Color perspective>
    std::int32_t get_result(int piece_count) const
    {
        constexpr Color us = perspective;
        constexpr Color them = !perspective;

        const auto &acc_us = accumulator.template get_accumulator<us>();
        const auto &acc_them = accumulator.template get_accumulator<them>();

        std::array<std::int8_t, NAccumulator> l0{};
        compute_l0(acc_us, acc_them, l0);

        const int bucket = bucket_for_piece_count(piece_count);
        const auto &ls = layer_stacks[bucket];

        const auto raw_l1 = ls.l1.get_raw(l0);

        std::array<std::int8_t, 2 * L2> l1_out{};
        for (int i = 0; i < L2; ++i)
        {
            std::int32_t x = raw_l1[i] >> WeightScaleBits;
            std::int32_t sqr = (x * x) >> 7;
            l1_out[i] = static_cast<std::int8_t>(std::clamp(sqr, 0, 127));
            l1_out[L2 + i] = static_cast<std::int8_t>(std::clamp(x, 0, 127));
        }
        const std::int32_t skip_raw = raw_l1[L2];

        std::array<std::int8_t, L3> l2_out{};
        ls.l2.process(l1_out, l2_out);

        const std::int32_t output_raw = ls.output.get_result(l2_out);

        const std::int64_t skip_rescaled = (static_cast<std::int64_t>(skip_raw) * SkipRescaleNum) / SkipRescaleDen;
        const std::int64_t layerstack_final = static_cast<std::int64_t>(output_raw) + skip_rescaled;

        const auto &psqt_us = psqt_accumulator.template get_accumulator<us>();
        const auto &psqt_them = psqt_accumulator.template get_accumulator<them>();
        const std::int64_t psqt_diff = static_cast<std::int64_t>(psqt_us[bucket]) - static_cast<std::int64_t>(psqt_them[bucket]);

        const std::int64_t combined = 2 * layerstack_final + psqt_diff;
        return static_cast<std::int32_t>(combined / FinalScale);
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        accumulator.template update_feature<activate, perspective>(feature);
        psqt_accumulator.template update_feature<activate, perspective>(feature);
    }
#ifdef CHESS26_UNIT_TESTING
    const auto &get_accumulator() const
    {
        return accumulator;
    }
#endif
};
