#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>

#include "core/piece/color.hpp"

// Tracks the PSQT (piece-square-table) output buckets that live alongside the
// main feature transformer accumulator. Quantized as int32 (scale =
// nnue2score * weight_scale_out), with an always-zero bias (see
// NNUEModel._init_psqt in nnue-pytorch: the psqt bias cancels out during
// perspective averaging, so it isn't stored).
template <int NFeatures, int NBuckets>
class PsqtAccumulatorLayer
{
    using AccTable = std::array<std::array<std::int32_t, NBuckets>, 2>;
    using WeightTable = std::array<std::array<std::int32_t, NBuckets>, NFeatures>;

    std::unique_ptr<AccTable> accumulators;
    std::unique_ptr<WeightTable> weights;

public:
    template <typename W>
    explicit PsqtAccumulatorLayer(W &&w)
    {
        weights = std::make_unique<WeightTable>(std::forward<W>(w));
        accumulators = std::make_unique<AccTable>();
        reset();
    }

    PsqtAccumulatorLayer(const PsqtAccumulatorLayer &other)
    {
        accumulators = std::make_unique<AccTable>(*other.accumulators);
        weights = std::make_unique<WeightTable>(*other.weights);
    }

    PsqtAccumulatorLayer &operator=(const PsqtAccumulatorLayer &other)
    {
        if (this != &other)
        {
            *accumulators = *other.accumulators;
            *weights = *other.weights;
        }
        return *this;
    }

    // Declared explicitly: the templated single-argument constructor above is
    // a forwarding reference and would otherwise hijack move-construction of
    // this class itself (it's a better match than the copy constructor for
    // an rvalue *this-typed argument).
    PsqtAccumulatorLayer(PsqtAccumulatorLayer &&other) noexcept = default;
    PsqtAccumulatorLayer &operator=(PsqtAccumulatorLayer &&other) noexcept = default;

    void reset()
    {
        (*accumulators)[WHITE].fill(0);
        (*accumulators)[BLACK].fill(0);
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        auto &acc = (*accumulators)[perspective];
        const auto &w_row = (*weights)[feature];

        for (int j = 0; j < NBuckets; ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

    template <Color perspective>
    const auto &get_accumulator() const
    {
        return (*accumulators)[perspective];
    }
};
