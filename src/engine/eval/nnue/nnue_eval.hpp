// nnue_eval.hpp
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/logger.hpp"
#include "common/fatal.hpp"
#include "common/constants.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include "features_encoder.hpp"
#include "core/piece/color.hpp"

// Features = 32 king buckets * 11 piece types * 64 squares (HalfKAv2_hm),
// matching features_encoder.hpp. Accum/PsqtBuckets/LsBuckets/L2/L3 match the
// architecture produced by nnue-pytorch commit 00bdf75 (--l1 256 --l2 32 --l3 32),
// which is the format version (0x7AF32F20) this project's .nnue file uses.
template <int Features = 22528, int Accum = 256, int NumPsqtBuckets = 8, int NumLsBuckets = 8, int L2 = 32, int L3 = 32>
class NnueEval
{
public:
    using Model = NnueModel<Features, Accum, NumPsqtBuckets, NumLsBuckets, L2, L3>;

private:
    Model model;

public:
    NnueEval(Model &&_model) : model(std::move(_model)) {}

    explicit NnueEval(const std::string &path)
        : model(load_model(path))
    {
    }

    // Returns the score from the side-to-move's own perspective (positive =
    // good for whoever is to move) — the network was trained with "us"/"them"
    // always meaning "side to move"/"opponent", so this must match the actual
    // side to move, not always WHITE.
    std::int32_t evaluate_abs(Color side_to_move, int piece_count) const
    {
        return side_to_move == WHITE
                   ? model.template get_result<WHITE>(piece_count)
                   : model.template get_result<BLACK>(piece_count);
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

    template <typename T>
    struct ElementCount
    {
        static constexpr std::size_t value = 1;
    };

    template <typename T, std::size_t N>
    struct ElementCount<std::array<T, N>>
    {
        static constexpr std::size_t value = N * ElementCount<T>::value;
    };

    static bool next_is_leb128_marker(std::ifstream &file)
    {
        constexpr std::string_view marker_text = "COMPRESSED_LEB128";
        constexpr std::streamsize marker_len = static_cast<std::streamsize>(marker_text.size());
        const auto marker_pos = tell(file);

        char marker[marker_len] = {};
        file.read(marker, marker_len);
        if (!file)
        {
            file.clear();
            file.seekg(marker_pos, std::ios::beg);
            return false;
        }

        const bool is_marker = (std::string(marker, marker_len) == marker_text);

        file.clear();
        file.seekg(marker_pos, std::ios::beg);
        check(file, "rewind after marker probe");

        return is_marker;
    }

    static std::vector<std::int64_t> decode_leb128_signed(
        const std::vector<std::uint8_t> &bytes,
        std::size_t expected_count)
    {
        std::vector<std::int64_t> out;
        out.reserve(expected_count);

        std::size_t k = 0;
        for (std::size_t i = 0; i < expected_count; ++i)
        {
            std::int64_t r = 0;
            int shift = 0;

            while (true)
            {
                if (k >= bytes.size())
                    FATAL("Unexpected end of compressed LEB128 stream");

                const std::uint8_t byte = bytes[k++];
                r |= (static_cast<std::int64_t>(byte & 0x7F) << shift);
                shift += 7;

                if ((byte & 0x80) == 0)
                {
                    const std::int64_t value =
                        ((byte & 0x40) == 0)
                            ? r
                            : (r | ~((static_cast<std::int64_t>(1) << shift) - 1));
                    out.push_back(value);
                    break;
                }

                if (shift >= 63)
                    FATAL("Invalid LEB128 value: too many continuation bytes");
            }
        }

        return out;
    }

    template <typename T>
    static void assign_flat_values(const std::vector<std::int64_t> &values, std::size_t &index, T &out)
    {
        out = static_cast<T>(values[index++]);
    }

    template <typename T, std::size_t N>
    static void assign_flat_values(const std::vector<std::int64_t> &values, std::size_t &index, std::array<T, N> &out)
    {
        for (auto &item : out)
            assign_flat_values(values, index, item);
    }

    template <typename T>
    static void read_tensor(std::ifstream &file, T &value, const std::string &label)
    {
        if (!next_is_leb128_marker(file))
        {
            read_binary(file, value);
            check(file, label);
            return;
        }

        constexpr std::string_view marker_text = "COMPRESSED_LEB128";
        constexpr std::streamsize marker_len = static_cast<std::streamsize>(marker_text.size());
        char marker[marker_len] = {};
        file.read(marker, marker_len);
        check(file, label + " marker");

        std::uint32_t compressed_len = 0;
        read_binary(file, compressed_len);
        check(file, label + " compressed_len");

        std::vector<std::uint8_t> bytes(compressed_len);
        file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(compressed_len));
        check(file, label + " compressed payload");

        constexpr std::size_t expected_count = ElementCount<T>::value;
        const auto decoded = decode_leb128_signed(bytes, expected_count);

        std::size_t idx = 0;
        assign_flat_values(decoded, idx, value);
    }

    static typename Model::L1Layer read_l1_layer(std::ifstream &file, const std::string &name)
    {
        return read_dense_layer<Accum, L2 + 1>(file, name);
    }

    template <int In, int Out>
    static DenseLayer<In, Out> read_dense_layer(std::ifstream &file, const std::string &name)
    {
        std::array<std::int32_t, Out> biases{};
        std::array<std::array<std::int8_t, In>, Out> weights{};

        read_binary(file, biases);
        check(file, name + " biases");

        read_binary(file, weights);
        check(file, name + " weights");

        return DenseLayer<In, Out>(std::move(weights), std::move(biases));
    }

    static Model load_model(const std::string &path)
    {
        logs::debug << "Loading NNUE..." << std::endl;

        std::ifstream file(path, std::ios::binary);
        if (!file)
            FATAL("Could not open NNUE file: " + path);

        const auto total_size = file_size(file);
        logs::debug << "[NNUE] file size = " << total_size << " bytes" << std::endl;

        // Heap-allocated: the feature transformer weight tensor (Features *
        // Accum * 2 bytes, ~11 MB for the real model) would overflow the
        // default thread stack if kept as a local std::array.
        std::array<std::int16_t, Accum> accumulator_biases{};
        auto accumulator_weights_ptr = std::make_unique<std::array<std::array<std::int16_t, Accum>, Features>>();
        auto &accumulator_weights = *accumulator_weights_ptr;
        auto psqt_weights_ptr = std::make_unique<std::array<std::array<std::int32_t, NumPsqtBuckets>, Features>>();
        auto &psqt_weights = *psqt_weights_ptr;

        std::uint32_t version = 0;
        std::uint32_t hash = 0;
        std::uint32_t description_len = 0;

        read_binary(file, version);
        check(file, "version");

        read_binary(file, hash);
        check(file, "hash");

        read_binary(file, description_len);
        check(file, "description_len");

        if (description_len > 1'000'000)
            FATAL("NNUE description_len looks invalid: " + std::to_string(description_len));

        std::string description(description_len, '\0');
        file.read(description.data(), description_len);
        check(file, "description");

        logs::debug << "[NNUE] description = " << description << std::endl;

        std::uint32_t ft_hash = 0;
        read_binary(file, ft_hash);
        check(file, "feature transformer hash");

        read_tensor(file, accumulator_biases, "accumulator_biases");
        read_tensor(file, accumulator_weights, "accumulator_weights");
        read_tensor(file, psqt_weights, "psqt_weights");

        logs::debug << "[NNUE] after feature transformer offset = " << tell(file) << std::endl;

        std::vector<typename Model::LayerStackBucket> buckets;
        buckets.reserve(NumLsBuckets);

        for (int b = 0; b < NumLsBuckets; ++b)
        {
            std::uint32_t fc_hash = 0;
            read_binary(file, fc_hash);
            check(file, "fc_hash bucket " + std::to_string(b));

            auto l1 = read_l1_layer(file, "l1 bucket " + std::to_string(b));
            auto l2 = read_dense_layer<2 * L2, L3>(file, "l2 bucket " + std::to_string(b));
            auto output = read_dense_layer<L3, 1>(file, "output bucket " + std::to_string(b));

            buckets.emplace_back(std::move(l1), std::move(l2), std::move(output));
        }

        const auto final_pos = tell(file);
        if (final_pos != total_size)
        {
            logs::debug << "[NNUE] WARNING: file not fully consumed. Remaining bytes = "
                        << (total_size - final_pos)
                        << std::endl;
        }

        return Model(
            std::move(accumulator_biases),
            std::move(accumulator_weights),
            std::move(psqt_weights),
            make_layer_stacks_array(buckets, std::make_index_sequence<NumLsBuckets>{}));
    }

    template <std::size_t... Is>
    static std::array<typename Model::LayerStackBucket, NumLsBuckets> make_layer_stacks_array(
        std::vector<typename Model::LayerStackBucket> &buckets,
        std::index_sequence<Is...>)
    {
        return {std::move(buckets[Is])...};
    }
};
