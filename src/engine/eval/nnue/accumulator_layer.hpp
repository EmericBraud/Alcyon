#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>

#include "core/piece/color.hpp"
#include "common/aligned_array.hpp"
#include "common/cpu.hpp"

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

    // Declared explicitly: a user-declared copy-ctor/assignment suppresses
    // the implicitly-generated move-ctor/assignment, which would otherwise
    // make std::move(AccumulatorLayer) silently fall back to the copy-ctor
    // above (an rvalue still binds to a const& parameter) instead of a real
    // move -- harmless now that copying is cheap (shared_ptr + a small
    // per-instance accumulator), but declaring these keeps that true even if
    // a future member is added that isn't cheap to copy.
    AccumulatorLayer(AccumulatorLayer &&) noexcept = default;
    AccumulatorLayer &operator=(AccumulatorLayer &&) noexcept = default;

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

    // Hints the weight row for `feature` into cache before it's actually
    // needed. Meant to be called several iterations ahead of the matching
    // update_feature() call in a caller's loop -- see initialize_perspective/
    // apply_threats_diff in nnue_eval.hpp for why "several" (a single
    // iteration of lookahead doesn't give the memory subsystem enough time
    // to complete the fetch before the row is actually read).
    void prefetch(int feature) const
    {
        cpu::prefetch<std::array<std::int16_t, NNeurons>, false, 0>(&(*weights)[feature]);
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