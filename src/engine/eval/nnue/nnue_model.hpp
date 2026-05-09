#pragma once

#include <array>
#include <cstdint>
#include <tuple>
#include <utility>
#include <type_traits>
#include <algorithm>

#include "common/constants.hpp"
#include "common/mask.hpp"
#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "engine/eval/nnue/features_encoder.hpp"
#include "engine/eval/nnue/accumulator_layer.hpp"
#include "engine/eval/nnue/dense_layer.hpp"

template <int...>
struct DenseLayerTuple;

template <int In, int Out, int... Rest>
struct DenseLayerTuple<In, Out, Rest...>
{
    using type = decltype(std::tuple_cat(
        std::declval<std::tuple<DenseLayer<In, Out>>>(),
        std::declval<typename DenseLayerTuple<Out, Rest...>::type>()));
};

template <int Last>
struct DenseLayerTuple<Last>
{
    using type = std::tuple<>;
};

template <int NFeatures, int NAccumulator, int... DenseLayers>
class NnueModel
{
    AccumulatorLayer<NFeatures, NAccumulator> accumulator;
    using LayersTuple =
        typename DenseLayerTuple<2 * NAccumulator, DenseLayers...>::type;

    LayersTuple dense_layers;

    template <std::size_t I, typename Input>
    auto forward_dense(const Input &input) const
    {
        auto &layer = std::get<I>(dense_layers);
        using Layer = std::remove_reference_t<decltype(layer)>;

        if constexpr (Layer::NNeurons == 1)
        {
            return layer.get_result(input);
        }
        else
        {
            std::array<std::int8_t, Layer::NNeurons> output{};
            layer.process(input, output);

            return forward_dense<I + 1>(output);
        }
    }

public:
    template <typename AccBiases, typename AccWeights, typename Layers>
    NnueModel(AccBiases &&acc_biases, AccWeights &&acc_weights, Layers &&layers)
        : accumulator(std::forward<AccBiases>(acc_biases), std::forward<AccWeights>(acc_weights)),
          dense_layers(std::forward<Layers>(layers))
    {
    }

    void initialize(const std::array<U64, constants::NumPieceVariants> &occupancies)
    {
        this->accumulator.reset();

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
            for (int piece_type = PAWN; piece_type <= QUEEN; ++piece_type)
            {
                U64 occupancy = occupancies[piece_type + piece_color * constants::PieceTypeCount];
                while (occupancy != 0ULL)
                {
                    int sq = cpu::pop_lsb(occupancy);
                    accumulator.template update_feature<true, WHITE>(feature_encoder::get_feature_index<WHITE>(white_king_sq, static_cast<Color>(piece_color), static_cast<Piece>(piece_type), sq));
                }
            }
        }
        for (int piece_color = WHITE; piece_color <= BLACK; ++piece_color)
        {
            for (int piece_type = PAWN; piece_type <= QUEEN; ++piece_type)
            {
                U64 occupancy = occupancies[piece_type + piece_color * constants::PieceTypeCount];
                while (occupancy != 0ULL)
                {
                    int sq = cpu::pop_lsb(occupancy);
                    accumulator.template update_feature<true, BLACK>(feature_encoder::get_feature_index<BLACK>(black_king_sq, static_cast<Color>(piece_color), static_cast<Piece>(piece_type), sq));
                }
            }
        }
    }

    template <Color perspective>
    std::int32_t get_result() const
    {
        std::array<std::int8_t, 2 * NAccumulator> input{};

        constexpr Color us = perspective;
        constexpr Color them = !perspective;

        const auto &acc_us = accumulator.template get_accumulator<us>();
        const auto &acc_them = accumulator.template get_accumulator<them>();

        for (int i = 0; i < NAccumulator; ++i)
        {
            input[i] =
                static_cast<std::int8_t>(std::clamp<std::int16_t>(acc_us[i], 0, 127));

            input[i + NAccumulator] =
                static_cast<std::int8_t>(std::clamp<std::int16_t>(acc_them[i], 0, 127));
        }

        return forward_dense<0>(input);
    }
    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        accumulator.template update_feature<activate, perspective>(feature);
    }
#ifdef CHESS26_UNIT_TESTING
    const auto &get_accumulator() const
    {
        return accumulator;
    }
#endif
};
