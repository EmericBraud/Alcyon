// dense_layer.hpp
#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <utility>

#include "common/aligned_array.hpp"
#include "dot_product.hpp"

// WeightScaleBits_: hidden-layer weights are quantized with scale 2^WeightScaleBits_
// (64 by default, matching nnue-pytorch/Stockfish quantization). Set to 0 for
// unquantized/identity test weights.
//
// Inputs are uint8_t activations in [0, 127] (every activation feeding a
// dense layer -- pairwise-square, squared-CReLU, ClippedReLU -- clamps to
// that range); weights are int8_t. That contract is what lets the inner
// loops run on nnue::dot::dot_u8_i8's unsigned x signed fast paths (see
// dot_product.hpp).
template <int NInputs_, int NNeurons_, int WeightScaleBits_ = 6>
class DenseLayer
{
public:
    static constexpr int NInputs = NInputs_;
    static constexpr int NNeurons = NNeurons_;

private:
    // weights/biases are read-only after construction and identical across
    // every thread's copy of this layer (one per layer-stack bucket, shared
    // by every VBoard/SearchWorker) -- shared_ptr<const T> means copying a
    // DenseLayer (e.g. via VBoard's copy ctor, done once per search thread)
    // just bumps a refcount instead of duplicating the underlying arrays.
    using WeightTable = AlignedArray<std::array<std::int8_t, NInputs_>, NNeurons_>;
    using BiasTable = AlignedArray<std::int32_t, NNeurons_>;

    std::shared_ptr<const WeightTable> weights;
    std::shared_ptr<const BiasTable> biases;

    std::uint8_t relu(std::int32_t acc) const;

public:
    void process(
        const std::array<std::uint8_t, NInputs_> &inputs,
        std::array<std::uint8_t, NNeurons_> &output) const;

    std::int32_t get_result(const std::array<std::uint8_t, NInputs_> &inputs) const
        requires(NNeurons_ == 1);

    template <std::size_t NInputsA, std::size_t NInputsB>
    std::int32_t get_result_split(
        const std::array<std::uint8_t, NInputsA> &inputs_a,
        const std::array<std::uint8_t, NInputsB> &inputs_b) const
        requires(NNeurons_ == 1 && NInputsA + NInputsB == static_cast<std::size_t>(NInputs_))
    {
        const std::int8_t *row = (*weights)[0].data();
        return (*biases)[0] +
               nnue::dot::dot_u8_i8<static_cast<int>(NInputsA)>(inputs_a.data(), row) +
               nnue::dot::dot_u8_i8<static_cast<int>(NInputsB)>(inputs_b.data(), row + NInputsA);
    }

    // Raw (unshifted, unclamped) bias + dot-product per neuron. Used by layers
    // whose activation isn't a plain ClippedReLU (e.g. the layer-stack L1,
    // which feeds a squared-CReLU and has an extra skip-connection output).
    std::array<std::int32_t, NNeurons_> get_raw(
        const std::array<std::uint8_t, NInputs_> &inputs) const;

    template <typename T, typename U>
    DenseLayer(T &&_weights, U &&_biases)
        : weights(std::make_shared<const WeightTable>(std::forward<T>(_weights))),
          biases(std::make_shared<const BiasTable>(std::forward<U>(_biases)))
    {
    }
};

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::uint8_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::relu(std::int32_t acc) const
{
    return static_cast<std::uint8_t>(std::clamp(acc >> WeightScaleBits_, 0, 127));
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline void DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::process(
    const std::array<std::uint8_t, NInputs_> &inputs,
    std::array<std::uint8_t, NNeurons_> &output) const
{
    for (int j = 0; j < NNeurons_; ++j)
    {
        const std::int32_t acc =
            (*biases)[j] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[j].data());
        output[j] = relu(acc);
    }
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::int32_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_result(
    const std::array<std::uint8_t, NInputs_> &inputs) const
    requires(NNeurons_ == 1)
{
    return (*biases)[0] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[0].data());
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::array<std::int32_t, NNeurons_> DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_raw(
    const std::array<std::uint8_t, NInputs_> &inputs) const
{
    std::array<std::int32_t, NNeurons_> output{};

    for (int j = 0; j < NNeurons_; ++j)
        output[j] = (*biases)[j] + nnue::dot::dot_u8_i8<NInputs_>(inputs.data(), (*weights)[j].data());

    return output;
}
