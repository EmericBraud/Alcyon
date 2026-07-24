// dense_layer.hpp
#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <utility>

// WeightScaleBits_: hidden-layer weights are quantized with scale 2^WeightScaleBits_
// (64 by default, matching nnue-pytorch/Stockfish quantization). Set to 0 for
// unquantized/identity test weights.
template <int NInputs_, int NNeurons_, int WeightScaleBits_ = 6>
class DenseLayer
{
public:
    static constexpr int NInputs = NInputs_;
    static constexpr int NNeurons = NNeurons_;

private:
    alignas(64) std::array<std::array<std::int8_t, NInputs_>, NNeurons_> weights;
    alignas(64) std::array<std::int32_t, NNeurons_> biases;

    std::int8_t relu(std::int32_t acc) const;

public:
    void process(
        const std::array<std::int8_t, NInputs_> &inputs,
        std::array<std::int8_t, NNeurons_> &output) const;

    std::int32_t get_result(const std::array<std::int8_t, NInputs_> &inputs) const
        requires(NNeurons_ == 1);

    template <std::size_t NInputsA, std::size_t NInputsB>
    std::int32_t get_result_split(
        const std::array<std::int8_t, NInputsA> &inputs_a,
        const std::array<std::int8_t, NInputsB> &inputs_b) const
        requires(NNeurons_ == 1 && NInputsA + NInputsB == static_cast<std::size_t>(NInputs_))
    {
        std::int32_t acc = biases[0];
        for (std::size_t i = 0; i < NInputsA; ++i)
            acc += static_cast<std::int32_t>(inputs_a[i]) * weights[0][i];
        for (std::size_t i = 0; i < NInputsB; ++i)
            acc += static_cast<std::int32_t>(inputs_b[i]) * weights[0][NInputsA + i];
        return acc;
    }

    // Raw (unshifted, unclamped) bias + dot-product per neuron. Used by layers
    // whose activation isn't a plain ClippedReLU (e.g. the layer-stack L1,
    // which feeds a squared-CReLU and has an extra skip-connection output).
    std::array<std::int32_t, NNeurons_> get_raw(
        const std::array<std::int8_t, NInputs_> &inputs) const;

    template <typename T, typename U>
    DenseLayer(T &&_weights, U &&_biases) : weights(std::forward<T>(_weights)), biases(std::forward<U>(_biases))
    {
    }
};

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::int8_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::relu(std::int32_t acc) const
{
    return static_cast<std::int8_t>(std::clamp(acc >> WeightScaleBits_, 0, 127));
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline void DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::process(
    const std::array<std::int8_t, NInputs_> &inputs,
    std::array<std::int8_t, NNeurons_> &output) const
{
    for (int j = 0; j < NNeurons_; ++j)
    {
        std::int32_t acc = biases[j];

        for (int i = 0; i < NInputs_; ++i)
            acc += static_cast<std::int32_t>(inputs[i]) * weights[j][i];

        output[j] = relu(acc);
    }
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::int32_t DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_result(
    const std::array<std::int8_t, NInputs_> &inputs) const
    requires(NNeurons_ == 1)
{
    std::int32_t acc = biases[0];

    for (int i = 0; i < NInputs_; ++i)
        acc += static_cast<std::int32_t>(inputs[i]) * weights[0][i];

    return acc;
}

template <int NInputs_, int NNeurons_, int WeightScaleBits_>
inline std::array<std::int32_t, NNeurons_> DenseLayer<NInputs_, NNeurons_, WeightScaleBits_>::get_raw(
    const std::array<std::int8_t, NInputs_> &inputs) const
{
    std::array<std::int32_t, NNeurons_> output{};

    for (int j = 0; j < NNeurons_; ++j)
    {
        std::int32_t acc = biases[j];

        for (int i = 0; i < NInputs_; ++i)
            acc += static_cast<std::int32_t>(inputs[i]) * weights[j][i];

        output[j] = acc;
    }

    return output;
}