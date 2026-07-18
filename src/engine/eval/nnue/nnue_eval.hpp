// nnue_eval.hpp
#pragma once

// Loader + evaluation facade for the NNUE format: Full_Threats (60,720
// features) + HalfKAv2_hm^ (22,528 real features after coalescing) combined
// feature set, L1=1024, L2=32, L3=32, 8 PSQT buckets, 8 layer-stack buckets.
// Produced by nnue-pytorch commit 4289208fe20cc6ec8753e5ee14c2f210de783ff0
// with default hyperparameters (model/config.py, model/modules/config.py).
//
// Verified against data/nnue/v2.nnue's actual header bytes (not just assumed):
//   - version == 0x6A448AFA (model/utils/serialize.py's hardcoded VERSION),
//     matches the file's first 4 bytes exactly.
//   - The header hash (bytes 4..8) equals fc_hash(L1=1024,L2=32,L3=32) XOR
//     ft_hash, where ft_hash (bytes 96..100, right after the description) is
//     independently readable from the file. Both fc_hash (which only depends
//     on L1/L2/L3, not on num_ls_buckets) and the combined feature_hash
//     (Full_Threats.HASH=0x8F234CB8, HalfKAv2_hm^.HASH=0x7F234CB8, combined via
//     ComposedFeatureTransformer._compute_hash) were computed independently in
//     Python and matched the file's bytes exactly. This confirms L1/L2/L3 and
//     the feature-set combination, but *not* num_psqt_buckets/num_ls_buckets
//     (those don't affect the hash) -- those are assumed to be the NNUEModel
//     defaults (8/8), consistent with everything else about this file.
//
// initialize() performs a full accumulator recompute from a board position;
// VBoard uses it only on load and on king moves. Non-king moves are handled
// incrementally via update_halfka_piece() (single feature per moved/captured
// piece) and collect_threats_scoped()/apply_threats_diff() (see
// full_threats_incremental.hpp for the Full_Threats scoped-recompute
// design) -- see virtual_board.hpp.

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common/logger.hpp"
#include "common/fatal.hpp"
#include "common/cpu.hpp"
#include "core/board/board.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include "engine/eval/nnue/full_threats_encoder.hpp"
#include "engine/eval/nnue/full_threats_incremental.hpp"
#include "engine/eval/nnue/halfka_v2_hm_encoder.hpp"

namespace nnue
{
    constexpr int L1 = 1024;
    constexpr int L2 = 32;
    constexpr int L3 = 32;
    constexpr int NumPsqtBuckets = 8;
    constexpr int NumLsBuckets = 8;
    constexpr int NumFullThreatsFeatures = threats::NUM_INPUTS;      // 60,720
    constexpr int NumHalfkaFeatures = halfka::NUM_REAL_FEATURES;     // 22,528
    constexpr int NumFeatures = NumFullThreatsFeatures + NumHalfkaFeatures; // 83,248

    constexpr std::uint32_t VERSION = 0x6A448AFA;

    class NnueEval
    {
    public:
        using Model = NnueModel<NumFeatures, L1, NumPsqtBuckets, NumLsBuckets, L2, L3>;

    private:
        Model model;

    public:
        NnueEval(Model &&_model) : model(std::move(_model)) {}

        explicit NnueEval(const std::string &path)
            : model(load_model(path))
        {
        }

        // Full recompute (see scope note above): rebuilds both perspectives'
        // accumulators from scratch by scanning the whole board.
        void initialize(const Board &board)
        {
            model.reset();

            std::vector<int> white_threats;
            std::vector<int> black_threats;
            threats::fill_features<WHITE>(board, white_threats);
            threats::fill_features<BLACK>(board, black_threats);

            for (int idx : white_threats)
                model.template update_feature<true, WHITE>(idx);
            for (int idx : black_threats)
                model.template update_feature<true, BLACK>(idx);

            const int white_ksq = board.king_sq[WHITE];
            const int black_ksq = board.king_sq[BLACK];

            for (int c = 0; c < 2; ++c)
            {
                const Color color = static_cast<Color>(c);
                for (int pt = PAWN; pt <= KING; ++pt)
                {
                    const Piece piece_type = static_cast<Piece>(pt);
                    U64 bb = board.pieces_occ[get_piece_index(piece_type, color)];
                    while (bb)
                    {
                        const int sq = cpu::pop_lsb(bb);

                        const int w_idx = NumFullThreatsFeatures + halfka::feature_index<WHITE>(white_ksq, color, piece_type, sq);
                        model.template update_feature<true, WHITE>(w_idx);

                        const int b_idx = NumFullThreatsFeatures + halfka::feature_index<BLACK>(black_ksq, color, piece_type, sq);
                        model.template update_feature<true, BLACK>(b_idx);
                    }
                }
            }
        }

        std::int32_t evaluate_abs(Color side_to_move, int piece_count) const
        {
            return side_to_move == WHITE
                       ? model.template get_result<WHITE>(piece_count)
                       : model.template get_result<BLACK>(piece_count);
        }

        // Incremental HalfKAv2_hm^ update for a single piece (add or remove),
        // analogous to v1's per-piece feature update. Not valid for the piece
        // whose own move is a king move -- callers must fall back to
        // initialize() (full refresh) in that case, since a king move changes
        // every HalfKAv2_hm^ feature for that perspective (king square/bucket
        // is baked into every other piece's index).
        template <bool activate, Color perspective>
        void update_halfka_piece(int king_sq, Color piece_color, Piece piece_type, int piece_sq)
        {
            if (piece_type == NO_PIECE)
                return;
            const int idx = NumFullThreatsFeatures + halfka::feature_index<perspective>(king_sq, piece_color, piece_type, piece_sq);
            model.template update_feature<activate, perspective>(idx);
        }

        // Incremental Full_Threats update for a non-king move, zero-copy
        // variant: instead of requiring two full Board snapshots (which
        // costs a full struct copy plus a heap-allocated History deep-copy
        // per call), callers collect the scoped feature set directly from
        // the *live* board twice -- once just before Board::play()/unplay()
        // mutates it, once just after. `touched_squares` (from/to/en-passant-
        // capture square) only depends on the Move itself, not on the
        // board's current state, so both calls can safely target the same
        // live board object. See full_threats_incremental.hpp for the
        // scoped-recompute design/tradeoff, and VBoard::play/unplay for the
        // call sites.
        template <Color perspective>
        void collect_threats_scoped(const Board &board, const std::vector<int> &touched_squares, std::vector<int> &out) const
        {
            threats::collect_move_scoped_features<perspective>(board, touched_squares, out);
        }

        // Diffs two (unsorted, possibly-duplicated) feature-index vectors
        // collected via collect_threats_scoped() -- one from "before" the
        // move, one from "after" -- and applies the resulting add/remove set
        // to the accumulator. Whichever vector is passed as `old_idx` is
        // removed and whichever is passed as `new_idx` is added; VBoard::
        // unplay() relies on this to reverse play()'s update by swapping the
        // two (post-move state first, pre-move state second).
        template <Color perspective>
        void apply_threats_diff(std::vector<int> old_idx, std::vector<int> new_idx)
        {
            std::sort(old_idx.begin(), old_idx.end());
            old_idx.erase(std::unique(old_idx.begin(), old_idx.end()), old_idx.end());
            std::sort(new_idx.begin(), new_idx.end());
            new_idx.erase(std::unique(new_idx.begin(), new_idx.end()), new_idx.end());

            std::size_t i = 0, j = 0;
            while (i < old_idx.size() && j < new_idx.size())
            {
                if (old_idx[i] < new_idx[j])
                    model.template update_feature<false, perspective>(old_idx[i++]);
                else if (new_idx[j] < old_idx[i])
                    model.template update_feature<true, perspective>(new_idx[j++]);
                else
                {
                    ++i;
                    ++j;
                } // unchanged feature, skip
            }
            while (i < old_idx.size())
                model.template update_feature<false, perspective>(old_idx[i++]);
            while (j < new_idx.size())
                model.template update_feature<true, perspective>(new_idx[j++]);
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

        static void check(std::ifstream &file, const std::string &label)
        {
            if (!file)
                FATAL("NNUE v2 read failed after: " + label);
        }

        static bool next_is_leb128_marker(std::ifstream &file)
        {
            constexpr std::string_view marker_text = "COMPRESSED_LEB128";
            constexpr std::streamsize marker_len = static_cast<std::streamsize>(marker_text.size());
            const auto marker_pos = file.tellg();

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

        // Reads `count` flat scalar values of type T (int8/int16/int32) into
        // `out` (of type U, possibly wider -- e.g. widening int8 into int16
        // storage), transparently handling the "COMPRESSED_LEB128" marker.
        template <typename T, typename U>
        static void read_tensor_flat(std::ifstream &file, std::vector<U> &out, std::size_t count, const std::string &label)
        {
            out.resize(count);

            if (!next_is_leb128_marker(file))
            {
                std::vector<T> raw(count);
                file.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(count * sizeof(T)));
                check(file, label);
                for (std::size_t i = 0; i < count; ++i)
                    out[i] = static_cast<U>(raw[i]);
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

            const auto decoded = decode_leb128_signed(bytes, count);
            for (std::size_t i = 0; i < count; ++i)
                out[i] = static_cast<U>(decoded[i]);
        }

        template <typename Layer, int In, int Out>
        static Layer read_dense_layer(std::ifstream &file, const std::string &name)
        {
            std::array<std::int32_t, Out> biases{};
            std::array<std::array<std::int8_t, In>, Out> weights{};

            std::vector<std::int32_t> bias_flat;
            read_tensor_flat<std::int32_t>(file, bias_flat, Out, name + " biases");
            std::copy(bias_flat.begin(), bias_flat.end(), biases.begin());

            std::vector<std::int8_t> weight_flat;
            read_tensor_flat<std::int8_t>(file, weight_flat, static_cast<std::size_t>(In) * Out, name + " weights");
            for (int o = 0; o < Out; ++o)
                for (int i = 0; i < In; ++i)
                    weights[o][i] = weight_flat[static_cast<std::size_t>(o) * In + i];

            return Layer(std::move(weights), std::move(biases));
        }

        static Model load_model(const std::string &path)
        {
            logs::debug << "Loading NNUE v2..." << std::endl;

            std::ifstream file(path, std::ios::binary);
            if (!file)
                FATAL("Could not open NNUE v2 file: " + path);

            std::uint32_t version = 0;
            std::uint32_t hash = 0;
            std::uint32_t description_len = 0;

            read_binary(file, version);
            check(file, "version");
            if (version != VERSION)
                FATAL("NNUE v2 file has unexpected version: " + std::to_string(version));

            read_binary(file, hash);
            check(file, "hash");

            read_binary(file, description_len);
            check(file, "description_len");
            if (description_len > 1'000'000)
                FATAL("NNUE v2 description_len looks invalid: " + std::to_string(description_len));

            std::string description(description_len, '\0');
            file.read(description.data(), description_len);
            check(file, "description");
            logs::debug << "[NNUE v2] description = " << description << std::endl;

            std::uint32_t ft_hash = 0;
            read_binary(file, ft_hash);
            check(file, "feature transformer hash");

            // Accumulator (feature transformer) bias: L1 int16 values.
            auto accumulator_biases_ptr = std::make_unique<std::array<std::int16_t, L1>>();
            {
                std::vector<std::int16_t> flat;
                read_tensor_flat<std::int16_t>(file, flat, L1, "accumulator biases");
                std::copy(flat.begin(), flat.end(), accumulator_biases_ptr->begin());
            }

            // Feature transformer weights, written as two independent segments
            // (Full_Threats then HalfKAv2_hm^, matching the "Full_Threats+HalfKAv2_hm^"
            // feature-name split order): each segment has its own weight tensor
            // (int8 for Full_Threats, int16 for HalfKAv2_hm^) and its own
            // int32 PSQT tensor.
            auto accumulator_weights_ptr = std::make_unique<std::array<std::array<std::int16_t, L1>, NumFeatures>>();
            auto psqt_weights_ptr = std::make_unique<std::array<std::array<std::int32_t, NumPsqtBuckets>, NumFeatures>>();
            auto &accumulator_weights = *accumulator_weights_ptr;
            auto &psqt_weights = *psqt_weights_ptr;

            auto read_segment = [&](int row_offset, int n_rows, bool weight_is_int8)
            {
                std::vector<std::int16_t> weight_flat;
                if (weight_is_int8)
                    read_tensor_flat<std::int8_t>(file, weight_flat, static_cast<std::size_t>(n_rows) * L1, "ft weight (int8 segment)");
                else
                    read_tensor_flat<std::int16_t>(file, weight_flat, static_cast<std::size_t>(n_rows) * L1, "ft weight (int16 segment)");

                for (int r = 0; r < n_rows; ++r)
                    for (int c = 0; c < L1; ++c)
                        accumulator_weights[row_offset + r][c] = weight_flat[static_cast<std::size_t>(r) * L1 + c];

                std::vector<std::int32_t> psqt_flat;
                read_tensor_flat<std::int32_t>(file, psqt_flat, static_cast<std::size_t>(n_rows) * NumPsqtBuckets, "ft psqt weight");
                for (int r = 0; r < n_rows; ++r)
                    for (int c = 0; c < NumPsqtBuckets; ++c)
                        psqt_weights[row_offset + r][c] = psqt_flat[static_cast<std::size_t>(r) * NumPsqtBuckets + c];
            };

            read_segment(0, NumFullThreatsFeatures, /*weight_is_int8=*/true);
            read_segment(NumFullThreatsFeatures, NumHalfkaFeatures, /*weight_is_int8=*/false);

            std::vector<typename Model::LayerStackBucket> buckets;
            buckets.reserve(NumLsBuckets);

            for (int b = 0; b < NumLsBuckets; ++b)
            {
                std::uint32_t fc_hash = 0;
                read_binary(file, fc_hash);
                check(file, "fc_hash bucket " + std::to_string(b));

                auto l1 = read_dense_layer<typename Model::L1Layer, L1, L2>(file, "l1 bucket " + std::to_string(b));
                auto l2 = read_dense_layer<typename Model::L2Layer, 2 * L2, L3>(file, "l2 bucket " + std::to_string(b));
                auto output = read_dense_layer<typename Model::OutputLayer, 2 * L2 + 2 * L3, 1>(file, "output bucket " + std::to_string(b));

                buckets.emplace_back(std::move(l1), std::move(l2), std::move(output));
            }

            return Model(
                std::move(*accumulator_biases_ptr),
                std::move(*accumulator_weights_ptr),
                std::move(*psqt_weights_ptr),
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
}
