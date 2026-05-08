// dense_layer.hpp
#pragma once

#include <array>
#include <cstdint>
#include <algorithm>
#include <utility>

template <int NInputs_, int NNeurons_>
class DenseLayer
{
public:
    static constexpr int NInputs = NInputs_;
    static constexpr int NNeurons = NNeurons_;

private:
    alignas(64) std::array<std::array<std::int8_t, NInputs_>, NNeurons_> weights;
    alignas(64) std::array<std::int32_t, NNeurons_> biases;

    std::int8_t relu(std::int32_t acc);

public:
    void process(
        const std::array<std::int8_t, NInputs_> &inputs,
        std::array<std::int8_t, NNeurons_> &output);

    std::int32_t get_result(const std::array<std::int8_t, NInputs_> &inputs)
        requires(NNeurons_ == 1);

    template <typename T, typename U>
    DenseLayer(T &&_weights, U &&_biases) : weights(std::forward<T>(_weights)), biases(std::forward<U>(_biases))
    {
    }
};

template <int NInputs_, int NNeurons_>
inline std::int8_t DenseLayer<NInputs_, NNeurons_>::relu(std::int32_t acc)
{
    return static_cast<std::int8_t>(std::clamp(acc, 0, 127));
}

template <int NInputs_, int NNeurons_>
inline void DenseLayer<NInputs_, NNeurons_>::process(
    const std::array<std::int8_t, NInputs_> &inputs,
    std::array<std::int8_t, NNeurons_> &output)
{
    for (int j = 0; j < NNeurons_; ++j)
    {
        std::int32_t acc = biases[j];

        for (int i = 0; i < NInputs_; ++i)
            acc += static_cast<std::int32_t>(inputs[i]) * weights[j][i];

        output[j] = relu(acc);
    }
}

template <int NInputs_, int NNeurons_>
inline std::int32_t DenseLayer<NInputs_, NNeurons_>::get_result(
    const std::array<std::int8_t, NInputs_> &inputs)
    requires(NNeurons_ == 1)
{
    std::int32_t acc = biases[0];

    for (int i = 0; i < NInputs_; ++i)
        acc += static_cast<std::int32_t>(inputs[i]) * weights[0][i];

    return acc;
}