#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>

#include "core/piece/color.hpp"

template <int NFeatures, int NNeurons>
class AccumulatorLayer
{
    std::unique_ptr<std::array<std::array<std::int16_t, NNeurons>, 2>> accumulators;
    std::unique_ptr<std::array<std::int16_t, NNeurons>> biases;
    std::unique_ptr<std::array<std::array<std::int8_t, NNeurons>, NFeatures>> weights;

public:
    template <typename B, typename W>
    AccumulatorLayer(B &&b, W &&w)
        : biases(std::forward<B>(b)),
          weights(std::forward<W>(w))
    {
        reset();
    }

    void reset()
    {
        accumulators[WHITE] = biases; // white perspective
        accumulators[BLACK] = biases; // black perspective
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        auto &acc = accumulators[perspective];

        for (int j = 0; j < NNeurons; ++j)
        {
            if constexpr (activate)
                acc[j] += weights[feature][j];
            else
                acc[j] -= weights[feature][j];
        }
    }

    template <bool activate, Color perspective, int nFeature>
    void update_feature()
    {
        static_assert(nFeature >= 0, "Feature index must be non-negative");
        static_assert(nFeature < NFeatures, "Feature out of index bounds");
        auto &acc = accumulators[perspective];
        for (int j = 0; j < NNeurons; ++j)
        {
            if constexpr (activate)
                acc[j] += this->weights[nFeature][j];
            else
                acc[j] -= this->weights[nFeature][j];
        }
    };

    const auto &get_accumulator(int perspective) const
    {
        return accumulators[perspective];
    }
    template <Color perspective>
    const auto &get_accumulator() const
    {
        return accumulators[perspective];
    }
};