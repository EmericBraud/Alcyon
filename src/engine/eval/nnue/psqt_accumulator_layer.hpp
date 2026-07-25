#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>

#include "core/piece/color.hpp"
#include "common/aligned_array.hpp"
#include "common/cpu.hpp"

// Tracks the PSQT (piece-square-table) output buckets that live alongside the
// main feature transformer accumulator. Quantized as int32 (scale =
// nnue2score * weight_scale_out), with an always-zero bias (see
// NNUEModel._init_psqt in nnue-pytorch: the psqt bias cancels out during
// perspective averaging, so it isn't stored).
template <int NFeatures, int NBuckets>
class PsqtAccumulatorLayer
{
public:
    // Public -- see AccumulatorLayer::AccTable's note.
    using AccTable = AlignedArray<std::array<std::int32_t, NBuckets>, 2>;

private:
    // AlignedArray: see common/aligned_array.hpp -- forces cache-line
    // alignment on the actual heap-allocated table (not just the pointer to it).
    using WeightTable = AlignedArray<std::array<std::int32_t, NBuckets>, NFeatures>;

    std::unique_ptr<AccTable> accumulators;
    std::shared_ptr<const WeightTable> weights;

public:
    template <typename W>
    explicit PsqtAccumulatorLayer(W &&w)
    {
        weights = std::make_shared<const WeightTable>(std::forward<W>(w));
        accumulators = std::make_unique<AccTable>();
        reset();
    }

    PsqtAccumulatorLayer(const PsqtAccumulatorLayer &other)
    {
        accumulators = std::make_unique<AccTable>(*other.accumulators);
        weights = other.weights;
    }

    PsqtAccumulatorLayer &operator=(const PsqtAccumulatorLayer &other)
    {
        if (this != &other)
        {
            *accumulators = *other.accumulators;
            weights = other.weights;
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

    template <Color perspective>
    void reset_perspective()
    {
        (*accumulators)[perspective].fill(0);
    }

    // See AccumulatorLayer::prefetch (accumulator_layer.hpp) -- same
    // rationale, called from the same call sites, kept for symmetry even
    // though this row is tiny (NBuckets * int32) compared to the main
    // feature-transformer row.
    void prefetch(int feature) const
    {
        cpu::prefetch<std::array<std::int32_t, NBuckets>, false, 0>(&(*weights)[feature]);
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

    // See AccumulatorLayer::raw_state/restore_raw_state -- same rationale.
    const AccTable &raw_state() const
    {
        return *accumulators;
    }
    void restore_raw_state(const AccTable &saved)
    {
        *accumulators = saved;
    }
};
