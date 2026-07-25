#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>

#include "core/piece/color.hpp"
#include "common/aligned_array.hpp"

template <int NFeatures, int NNeurons>
class AccumulatorLayer
{
    // On utilise des alias pour rendre le code lisible
    using AccTable = AlignedArray<std::array<std::int16_t, NNeurons>, 2>;
    using BiasTable = AlignedArray<std::int16_t, NNeurons>;
    // Feature transformer weights are quantized as int16 (scale = 127), not int8.
    using WeightTable = AlignedArray<std::array<std::int16_t, NNeurons>, NFeatures>;

    std::unique_ptr<AccTable> accumulators;
    std::shared_ptr<const BiasTable> biases;
    std::shared_ptr<const WeightTable> weights;

public:
    template <typename B, typename W>
    AccumulatorLayer(B &&b, W &&w)
    {
        // Allocation sur le tas et copie des données initiales
        biases = std::make_shared<const BiasTable>(std::forward<B>(b));
        weights = std::make_shared<const WeightTable>(std::forward<W>(w));
        accumulators = std::make_unique<AccTable>();
        reset();
    }
    AccumulatorLayer(const AccumulatorLayer &other)
    {
        accumulators = std::make_unique<AccTable>(*other.accumulators);
        biases = other.biases;
        weights = other.weights;
    }

    AccumulatorLayer &operator=(const AccumulatorLayer &other)
    {
        if (this != &other)
        {
            *accumulators = *other.accumulators;
            biases = other.biases;
            weights = other.weights;
        }
        return *this;
    }

    void reset()
    {
        // On copie les biais dans les deux accumulateurs (perspective Blanche et Noire)
        (*accumulators)[WHITE] = *biases;
        (*accumulators)[BLACK] = *biases;
    }

    // Resets only one perspective's accumulator -- used when a king move
    // only invalidates that perspective's features (see
    // NnueEval::initialize_perspective()), leaving the other perspective's
    // accumulator untouched.
    template <Color perspective>
    void reset_perspective()
    {
        (*accumulators)[perspective] = *biases;
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        // On déréférence d'abord le pointeur (*accumulators)
        auto &acc = (*accumulators)[perspective];
        const auto &w_row = (*weights)[feature];

        for (int j = 0; j < NNeurons; ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

    // Version avec template pour l'index de feature
    template <bool activate, Color perspective, int nFeature>
    void update_feature()
    {
        static_assert(nFeature >= 0 && nFeature < NFeatures, "Feature index out of bounds");

        auto &acc = (*accumulators)[perspective];
        const auto &w_row = (*weights)[nFeature];

        for (int j = 0; j < NNeurons; ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

    const auto &get_accumulator(int perspective) const
    {
        return (*accumulators)[perspective];
    }

    template <Color perspective>
    const auto &get_accumulator() const
    {
        return (*accumulators)[perspective];
    }
};