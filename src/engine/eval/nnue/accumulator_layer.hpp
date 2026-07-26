#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <memory>
#include <experimental/simd>

#include "core/piece/color.hpp"
#include "common/aligned_array.hpp"
#include "common/cpu.hpp"
#include "common/simd.hpp"

// NThreatFeatures: rows [0, NThreatFeatures) are stored as int8 (Full_Threats
// segment of the combined feature set, see nnue_eval.hpp) instead of int16 --
// those rows are ~half the file's weight bytes (170MB -> ~108MB: 60,720 rows
// at 1024 * 1 byte instead of 1024 * 2 bytes) and, per profiling, are nearly
// all of the threat-diff apply pass's DRAM traffic (the halfka segment updates via a
// different, much colder path -- HalfKA piece toggles, one/two features per
// non-king move). Widening int8 -> int16 on the fly (sign-extend, like
// vpmovsxbw) costs a few cycles of ALU work that's fully hidden behind the
// cache-miss latency the load itself already pays, so this halves bytes
// fetched from DRAM per update with no change to the arithmetic result
// (weights are quantized losslessly into int8 by the same trainer/export
// pipeline that already stores Full_Threats as int8 in the file -- see
// nnue_eval.hpp's read_segment/weight_is_int8).
template <int NFeatures, int NNeurons, int NThreatFeatures>
class AccumulatorLayer
{
    static_assert(NThreatFeatures >= 0 && NThreatFeatures <= NFeatures,
                  "NThreatFeatures must be within [0, NFeatures]");

public:
    // Public so callers can snapshot/restore the full per-perspective state
    // (see raw_state/restore_raw_state below) without depending on its
    // internal layout beyond "one opaque, copyable blob".
    using AccTable = AlignedArray<std::array<std::int16_t, NNeurons>, 2>;

private:
    using BiasTable = AlignedArray<std::int16_t, NNeurons>;
    // Feature transformer weights: Full_Threats rows are stored int8 (see
    // class comment above), HalfKAv2_hm^ rows stay int16 (scale = 127) --
    // matching the file's own per-segment encoding, so no widening happens
    // at load time (nnue_eval.hpp's read_segment reads each segment directly
    // into its native width).
    using Int8WeightTable = AlignedArray<std::array<std::int8_t, NNeurons>, NThreatFeatures>;
    using Int16WeightTable = AlignedArray<std::array<std::int16_t, NNeurons>, NFeatures - NThreatFeatures>;

    std::unique_ptr<AccTable> accumulators;
    std::shared_ptr<const BiasTable> biases;
    std::shared_ptr<const Int8WeightTable> threat_weights;
    std::shared_ptr<const Int16WeightTable> halfka_weights;

    // HalfKAv2_hm^ rows are already int16 -- straight vectorizable add/sub,
    // same as before the int8 split.
    template <bool activate>
    static void apply_row(std::array<std::int16_t, NNeurons> &acc, const std::array<std::int16_t, NNeurons> &w_row)
    {
        for (int j = 0; j < NNeurons; ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

    // Full_Threats rows are int8 in memory (half the bytes fetched from DRAM
    // per update, see class comment above): each chunk of int16_v::size()
    // int8 weights is loaded and widened (sign-extended, like vpmovsxbw) to
    // int16 lanes before adding/subtracting -- the widen is pure ALU work
    // that overlaps with the load's cache-miss latency, so it costs nothing
    // extra versus a native int16 row on the memory-bound hot path this
    // serves (NnueEval::apply_list/initialize_perspective).
    template <bool activate>
    static void apply_row(std::array<std::int16_t, NNeurons> &acc, const std::array<std::int8_t, NNeurons> &w_row)
    {
        using int8_n = stdx::fixed_size_simd<std::int8_t, simd::SimdSize16>;

        std::size_t j = 0;
        for (; j + simd::SimdSize16 <= static_cast<std::size_t>(NNeurons); j += simd::SimdSize16)
        {
            int8_n raw;
            raw.copy_from(w_row.data() + j, stdx::element_aligned);
            const simd::int16_v widened = stdx::simd_cast<simd::int16_v>(raw);

            simd::int16_v acc_v;
            acc_v.copy_from(acc.data() + j, stdx::element_aligned);
            if constexpr (activate)
                acc_v += widened;
            else
                acc_v -= widened;
            acc_v.copy_to(acc.data() + j, stdx::element_aligned);
        }
        for (; j < static_cast<std::size_t>(NNeurons); ++j)
        {
            if constexpr (activate)
                acc[j] += w_row[j];
            else
                acc[j] -= w_row[j];
        }
    }

public:
    template <typename B, typename W8, typename W16>
    AccumulatorLayer(B &&b, W8 &&w8, W16 &&w16)
    {
        // Allocation sur le tas et copie des données initiales
        biases = std::make_shared<const BiasTable>(std::forward<B>(b));
        threat_weights = std::make_shared<const Int8WeightTable>(std::forward<W8>(w8));
        halfka_weights = std::make_shared<const Int16WeightTable>(std::forward<W16>(w16));
        accumulators = std::make_unique<AccTable>();
        reset();
    }
    AccumulatorLayer(const AccumulatorLayer &other)
    {
        accumulators = std::make_unique<AccTable>(*other.accumulators);
        biases = other.biases;
        threat_weights = other.threat_weights;
        halfka_weights = other.halfka_weights;
    }

    AccumulatorLayer &operator=(const AccumulatorLayer &other)
    {
        if (this != &other)
        {
            *accumulators = *other.accumulators;
            biases = other.biases;
            threat_weights = other.threat_weights;
            halfka_weights = other.halfka_weights;
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
    // NnueEval::apply_list in nnue_eval.hpp for why "several" (a single
    // iteration of lookahead doesn't give the memory subsystem enough time
    // to complete the fetch before the row is actually read).
    void prefetch(int feature) const
    {
        if (feature < NThreatFeatures)
            cpu::prefetch<std::array<std::int8_t, NNeurons>, false, 0>(&(*threat_weights)[feature]);
        else
            cpu::prefetch<std::array<std::int16_t, NNeurons>, false, 0>(&(*halfka_weights)[feature - NThreatFeatures]);
    }

    template <bool activate, Color perspective>
    void update_feature(int feature)
    {
        // On déréférence d'abord le pointeur (*accumulators)
        auto &acc = (*accumulators)[perspective];

        if (feature < NThreatFeatures)
            apply_row<activate>(acc, (*threat_weights)[feature]);
        else
            apply_row<activate>(acc, (*halfka_weights)[feature - NThreatFeatures]);
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

    // Raw access to both perspectives' state, for saving/restoring a full
    // snapshot (see NnueEval::push_state/pop_state) -- much cheaper than
    // replaying the incremental feature updates when unwinding a move.
    const AccTable &raw_state() const
    {
        return *accumulators;
    }
    void restore_raw_state(const AccTable &saved)
    {
        *accumulators = saved;
    }
};