// nnue_eval.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>

#include "common/logger.hpp"
#include "common/fatal.hpp"
#include "common/constants.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include "features_encoder.hpp"
#include "core/piece/color.hpp"

template <int Features = 24576, int Accum = 256, int Layer1Dims = 32, int Layer2Dims = 32, int Layer3Dims = 1>
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

    std::int32_t evaluate_abs() const
    {
        return model.template get_result<WHITE>();
    }

    void initialize(const std::array<U64, constants::NumPieceVariants> &occupancies)
    {
        model.initialize(occupancies);
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
    template <typename T>
    static void read_binary(std::ifstream &file, T &value)
    {
        file.read(reinterpret_cast<char *>(&value), sizeof(T));
    }

    static std::streamoff tell(std::ifstream &file)
    {
        return file.tellg();
    }

    static std::streamoff file_size(std::ifstream &file)
    {
        auto cur = file.tellg();
        file.seekg(0, std::ios::end);
        auto end = file.tellg();
        file.seekg(cur, std::ios::beg);
        return end;
    }

    static void check(std::ifstream &file, const std::string &label)
    {
        if (!file)
            FATAL("NNUE read failed after: " + label);
    }

    static Model load_model(const std::string &path)
    {
        logs::debug << "Loading NNUE..." << std::endl;

        std::ifstream file(path, std::ios::binary);
        if (!file)
            FATAL("Could not open NNUE file: " + path);

        const auto total_size = file_size(file);
        logs::debug << "[NNUE] file size = " << total_size << " bytes" << std::endl;

        std::array<std::int16_t, Accum> accumulator_biases{};
        std::array<std::array<std::int8_t, Accum>, Features> accumulator_weights{};

        std::uint32_t version = 0;
        std::uint32_t hash = 0;
        std::uint32_t description_len = 0;

        read_binary(file, version);
        check(file, "version");

        read_binary(file, hash);
        check(file, "hash");

        read_binary(file, description_len);
        check(file, "description_len");

        logs::debug << "[NNUE] version = 0x" << std::hex << version << std::dec << std::endl;
        logs::debug << "[NNUE] hash = 0x" << std::hex << hash << std::dec << std::endl;
        logs::debug << "[NNUE] description_len = " << description_len << std::endl;

        if (description_len > 1'000'000)
            FATAL("NNUE description_len looks invalid: " + std::to_string(description_len));

        std::string description(description_len, '\0');
        file.read(description.data(), description_len);
        check(file, "description");

        logs::debug << "[NNUE] description = " << description << std::endl;
        logs::debug << "[NNUE] after description offset = " << tell(file) << std::endl;

        std::uint32_t ft_hash = 0;
        read_binary(file, ft_hash);
        check(file, "feature transformer hash");

        logs::debug << "[NNUE] ft_hash = 0x" << std::hex << ft_hash << std::dec << std::endl;
        logs::debug << "[NNUE] after ft_hash offset = " << tell(file) << std::endl;

        read_binary(file, accumulator_biases);
        check(file, "accumulator_biases");

        logs::debug << "[NNUE] after accumulator_biases offset = " << tell(file) << std::endl;

        logs::debug << "[NNUE] first accumulator biases: ";
        for (int i = 0; i < std::min(10, Accum); ++i)
            logs::debug << accumulator_biases[i] << " ";
        logs::debug << std::endl;

        read_binary(file, accumulator_weights);
        check(file, "accumulator_weights");

        logs::debug << "[NNUE] after accumulator_weights offset = " << tell(file) << std::endl;

        logs::debug << "[NNUE] sample accumulator weights:" << std::endl;

        logs::debug << "  w[0][0..9] = ";
        for (int i = 0; i < std::min(10, Accum); ++i)
            logs::debug << static_cast<int>(accumulator_weights[0][i]) << " ";
        logs::debug << std::endl;

        logs::debug << "  w[1][0..9] = ";
        for (int i = 0; i < std::min(10, Accum); ++i)
            logs::debug << static_cast<int>(accumulator_weights[1][i]) << " ";
        logs::debug << std::endl;

        constexpr std::streamoff dense_bytes =
            4 + // one fc_hash
            sizeof(std::array<std::int32_t, Layer1Dims>) +
            sizeof(std::array<std::array<std::int8_t, 2 * Accum>, Layer1Dims>) +
            sizeof(std::array<std::int32_t, Layer2Dims>) +
            sizeof(std::array<std::array<std::int8_t, Layer1Dims>, Layer2Dims>) +
            sizeof(std::array<std::int32_t, Layer3Dims>) +
            sizeof(std::array<std::array<std::int8_t, Layer2Dims>, Layer3Dims>);

        const std::streamoff dense_start = total_size - dense_bytes;
        const std::streamoff current_pos = tell(file);

        logs::debug << "[NNUE] dense_bytes = " << dense_bytes << std::endl;
        logs::debug << "[NNUE] computed dense_start = " << dense_start << std::endl;
        logs::debug << "[NNUE] current offset before dense = " << current_pos << std::endl;

        if (current_pos > dense_start)
        {
            FATAL("NNUE parser consumed too much before dense layers. current=" +
                  std::to_string(current_pos) +
                  ", dense_start=" +
                  std::to_string(dense_start));
        }

        if (current_pos < dense_start)
        {
            logs::debug << "[NNUE] skipping unknown block before dense layers: "
                        << (dense_start - current_pos)
                        << " bytes"
                        << std::endl;

            file.seekg(dense_start, std::ios::beg);
            check(file, "seek to dense_start");
        }

        std::uint32_t fc_hash = 0;
        read_binary(file, fc_hash);
        check(file, "fc_hash");

        logs::debug << "[NNUE] fc_hash = 0x" << std::hex << fc_hash << std::dec << std::endl;
        logs::debug << "[NNUE] after fc_hash offset = " << tell(file) << std::endl;

        auto dense_layers = std::make_tuple(
            read_dense_layer_debug<2 * Accum, Layer1Dims>(file, "layer1"),
            read_dense_layer_debug<Layer1Dims, Layer2Dims>(file, "layer2"),
            read_dense_layer_debug<Layer2Dims, Layer3Dims>(file, "output"));

        check(file, "dense layers");

        const auto final_pos = tell(file);

        logs::debug << "[NNUE] final offset = "
                    << final_pos
                    << " / "
                    << total_size
                    << std::endl;

        if (final_pos != total_size)
        {
            logs::debug << "[NNUE] WARNING: file not fully consumed. Remaining bytes = "
                        << (total_size - final_pos)
                        << std::endl;
        }

        return Model(
            std::move(accumulator_biases),
            std::move(accumulator_weights),
            std::move(dense_layers));
    }

    template <int In, int Out>
    static DenseLayer<In, Out> read_dense_layer(std::ifstream &file)
    {
        std::array<std::int32_t, Out> biases{};
        std::array<std::array<std::int8_t, In>, Out> weights{};

        read_binary(file, biases);
        read_binary(file, weights);

        return DenseLayer<In, Out>(
            std::move(weights),
            std::move(biases));
    }

    template <int In, int Out>
    static DenseLayer<In, Out> read_dense_layer_debug(
        std::ifstream &file,
        const std::string &name)
    {
        logs::debug << "[NNUE] reading "
                    << name
                    << " at offset="
                    << tell(file)
                    << std::endl;

        std::array<std::int32_t, Out> biases{};
        std::array<std::array<std::int8_t, In>, Out> weights{};

        logs::debug << "[NNUE] "
                    << name
                    << " expected biases bytes = "
                    << sizeof(biases)
                    << ", weights bytes = "
                    << sizeof(weights)
                    << std::endl;

        read_binary(file, biases);
        check(file, name + " biases");

        logs::debug << "[NNUE] "
                    << name
                    << " after biases offset="
                    << tell(file)
                    << std::endl;

        read_binary(file, weights);
        check(file, name + " weights");

        logs::debug << "[NNUE] "
                    << name
                    << " after weights offset="
                    << tell(file)
                    << std::endl;

        logs::debug << "[NNUE] " << name << " first biases: ";
        for (int i = 0; i < std::min(10, Out); ++i)
            logs::debug << biases[i] << " ";
        logs::debug << std::endl;

        logs::debug << "[NNUE] " << name << " first weights row: ";
        for (int i = 0; i < std::min(10, In); ++i)
            logs::debug << static_cast<int>(weights[0][i]) << " ";
        logs::debug << std::endl;

        return DenseLayer<In, Out>(
            std::move(weights),
            std::move(biases));
    }
};