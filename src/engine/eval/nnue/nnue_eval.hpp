// nnue_eval.hpp
#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "common/fatal.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include "features_encoder.hpp"
#include "core/piece/color.hpp"

template <int Features = 40960, int Accum = 256, int Layer1Dims = 32, int Layer2Dims = 32, int Layer3Dims = 1>
class NnueEval
{
public:
    using Model = NnueModel<
        Features,
        Accum,
        Layer1Dims, Layer2Dims, Layer3Dims>;

private:
    Model model;

public:
    NnueEval(Model &&_model) : model(std::move(_model)) {}

    explicit NnueEval(const std::string &path)
        : model(load_model(path))
    {
    }

    std::int32_t evaluate_abs()
    {
        return model.template get_result<WHITE>();
    }

    template <bool activate, Color perspective>
    void update_feature(int feature_idx)
    {
        model.template update_feature<activate, perspective>(feature_idx);
    }
#ifdef CHESS26_UNIT_TESTING
    const auto &get_accumulator() const
    {
        return model.get_accumulator();
    }
#endif
private:
    static Model load_model(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);

        if (!file)
            FATAL("Could not open NNUE file: " + path);

        std::array<std::int16_t, Accum> accumulator_biases{};
        std::array<std::array<std::int8_t, Accum>, Features> accumulator_weights{};

        read_binary(file, accumulator_biases);
        read_binary(file, accumulator_weights);

        auto dense_layers = std::make_tuple(
            read_dense_layer<2 * Accum, Layer1Dims>(file),
            read_dense_layer<Layer1Dims, Layer2Dims>(file),
            read_dense_layer<Layer2Dims, Layer3Dims>(file));

        if (!file)
            FATAL("Invalid or truncated NNUE file: " + path);

        return Model(
            std::move(accumulator_biases),
            std::move(accumulator_weights),
            std::move(dense_layers));
    }

    template <typename T>
    static void read_binary(std::ifstream &file, T &value)
    {
        file.read(reinterpret_cast<char *>(&value), sizeof(T));
    }

    template <int In, int Out>
    static DenseLayer<In, Out> read_dense_layer(std::ifstream &file)
    {
        std::array<std::array<std::int8_t, In>, Out> weights{};
        std::array<std::int32_t, Out> biases{};

        read_binary(file, weights);
        read_binary(file, biases);

        return DenseLayer<In, Out>(
            std::move(weights),
            std::move(biases));
    }
};