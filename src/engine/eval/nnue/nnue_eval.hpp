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
#include <type_traits>
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

    // Software-prefetch lookahead for the feature-update loops below
    // (initialize_perspective/apply_threats_diff): a single iteration of
    // lookahead doesn't give the ~500-cycle DRAM round trip (measured on
    // this machine) enough time to complete before update_feature() actually
    // reads that row, since one update_feature() call only takes ~100-120
    // cycles with warm data. Tune empirically -- see the A/B methodology used
    // for the other NNUE fixes this session.
    constexpr int PrefetchDistance = 4;

    class NnueEval
    {
    public:
        using Model = NnueModel<NumFeatures, L1, NumPsqtBuckets, NumLsBuckets, L2, L3>;

    private:
        Model model;

        // Reusable, allocated-once membership-stamp buffer for
        // apply_threats_diff()'s dedup/diff (see below): sized to
        // Full_Threats' NUM_INPUTS so every raw feature index collected by
        // the scoped-recompute path (full_threats_incremental.hpp) can be
        // used directly as an index into it. Zero-initialized once here
        // (construction/load time, not per search node); apply_threats_diff
        // never clears it, it just bumps diff_generation so every stamp
        // from prior calls compares unequal to the current one.
        //
        // The three stamps for a given idx (old/new/emitted) are packed into
        // one struct (array-of-structs) rather than three separate
        // NUM_INPUTS-sized arrays (structure-of-arrays): apply_threats_diff
        // touches all three stamps for the same idx together, and idx values
        // are scattered across the full 60,720-entry range with no spatial
        // correlation, so the SoA layout meant each of those three per-idx
        // accesses could land on a different, likely-cold cache line. Packing
        // them together means one cache-line fetch serves all three reads/
        // writes for a given idx instead of up to three.
        struct ThreatMark
        {
            std::uint32_t old_gen = 0;
            std::uint32_t new_gen = 0;
            std::uint32_t emitted_gen = 0;
        };
        std::array<ThreatMark, threats::NUM_INPUTS> threat_marks{};
        std::uint32_t diff_generation = 0;

    public:
        NnueEval(Model &&_model) : model(std::move(_model)) {}

        explicit NnueEval(const std::string &path)
            : model(load_model(path))
        {
        }

        // Full recompute of a single perspective's accumulator contribution
        // (Full_Threats + HalfKA), scanning the whole board -- used by
        // initialize() below (both perspectives) and, on a king move, for
        // just the mover's own perspective (see VBoard::play/unplay): only
        // that perspective's HalfKA indices and Full_Threats orientation
        // depend on that perspective's own king square, so the other
        // perspective never needs this full rescan on a king move.
        template <Color perspective>
        void initialize_perspective(const Board &board)
        {
            model.template reset_perspective<perspective>();

            // Piece (HalfKA) indices are collected *before* fill_features()
            // below, deliberately -- each one is prefetched right as it's
            // discovered, so fill_features()'s own (non-trivial) scan time
            // covers that prefetch's latency for free, instead of the
            // piece-consumption loop having to wait PrefetchDistance
            // iterations with no lookahead at its start the way
            // perspective_threats necessarily does (its indices aren't known
            // until fill_features returns, so nothing can prefetch them
            // ahead of that).
            threats::FixedIntList<64> piece_features;
            const int ksq = board.king_sq[perspective];
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
                        const int idx = NumFullThreatsFeatures + halfka::feature_index<perspective>(ksq, color, piece_type, sq);
                        model.prefetch_feature(idx);
                        piece_features.push_back(idx);
                    }
                }
            }

            threats::FixedIntList<threats::MAX_FULL_SCAN_THREAT_FEATURES> perspective_threats;
            threats::fill_features<perspective>(board, perspective_threats);

            const int n_threats = perspective_threats.size();
            const int n_pieces = piece_features.size();
            for (int i = 0; i < n_threats; ++i)
            {
                // Once the lookahead window runs past the end of
                // perspective_threats, spill it into piece_features instead of
                // just stopping: those entries are already known (collected
                // above) so there's no reason to leave the last few threat
                // iterations uncovered when piece prefetches can fill the gap.
                const int lookahead = i + PrefetchDistance;
                if (lookahead < n_threats)
                    model.prefetch_feature(perspective_threats[lookahead]);
                else if (const int piece_idx = lookahead - n_threats; piece_idx < n_pieces)
                    model.prefetch_feature(piece_features[piece_idx]);
                model.template update_feature<true, perspective>(perspective_threats[i]);
            }

            for (int i = 0; i < n_pieces; ++i)
            {
                // Same spill idea in reverse doesn't apply here (nothing comes
                // after piece_features), but the threats loop above already
                // primed the first PrefetchDistance piece entries by the time
                // we get here, so this loop's own cold-start is the piece
                // count it spilled into, not PrefetchDistance from zero.
                if (i + PrefetchDistance < n_pieces)
                    model.prefetch_feature(piece_features[i + PrefetchDistance]);
                model.template update_feature<true, perspective>(piece_features[i]);
            }
        }

        // Full recompute (see scope note above): rebuilds both perspectives'
        // accumulators from scratch by scanning the whole board.
        void initialize(const Board &board)
        {
            initialize_perspective<WHITE>(board);
            initialize_perspective<BLACK>(board);
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
        void collect_threats_scoped(const Board &board, const threats::FixedIntList<threats::MAX_TOUCHED_SQUARES> &touched_squares, threats::FixedIntList<threats::MAX_THREAT_FEATURES> &out) const
        {
            threats::collect_move_scoped_features<perspective>(board, touched_squares, out);
        }

        // Combined-perspective variant: shares the magic-bitboard attacker/
        // defender scan between both perspectives instead of repeating it
        // once per perspective. See full_threats_incremental.hpp.
        void collect_threats_scoped_both(const Board &board, const threats::FixedIntList<threats::MAX_TOUCHED_SQUARES> &touched_squares, threats::FixedIntList<threats::MAX_THREAT_FEATURES> &out_white, threats::FixedIntList<threats::MAX_THREAT_FEATURES> &out_black) const
        {
            threats::collect_move_scoped_features_both(board, touched_squares, out_white, out_black);
        }

        // Diffs two (unsorted, possibly-duplicated) feature-index lists
        // collected via collect_threats_scoped() -- one from "before" the
        // move, one from "after" -- and applies the resulting add/remove set
        // to the accumulator. Whichever list is passed as `old_idx` is
        // removed and whichever is passed as `new_idx` is added; VBoard::
        // unplay() relies on this to reverse play()'s update by swapping the
        // two (post-move state first, pre-move state second).
        //
        // Zero heap allocation, zero std::sort/std::unique: membership in
        // each list is recorded via a per-call "generation" stamp in reusable
        // NUM_INPUTS-sized member arrays (allocated once, at construction --
        // not a per-call/per-search-node allocation), so both dedup and the
        // add/remove diff are done in O(old_idx.size() + new_idx.size())
        // without ever clearing the arrays themselves.
        template <Color perspective>
        void apply_threats_diff(const threats::FixedIntList<threats::MAX_THREAT_FEATURES> &old_idx, const threats::FixedIntList<threats::MAX_THREAT_FEATURES> &new_idx)
        {
            const std::uint32_t gen = ++diff_generation;

            for (int idx : old_idx)
                threat_marks[idx].old_gen = gen;
            for (int idx : new_idx)
                threat_marks[idx].new_gen = gen;

            // Prefetch-N-ahead (see PrefetchDistance above): the row for
            // idx[i + PrefetchDistance] is hinted regardless of whether that
            // entry turns out to be a duplicate skipped by the mark check
            // below -- a wasted prefetch is harmless, and most entries in
            // these lists do end up applied, so it isn't wasted often.
            const int n_old = old_idx.size();
            for (int i = 0; i < n_old; ++i)
            {
                if (i + PrefetchDistance < n_old)
                    model.prefetch_feature(old_idx[i + PrefetchDistance]);

                const int idx = old_idx[i];
                ThreatMark &mark = threat_marks[idx];
                if (mark.new_gen != gen && mark.emitted_gen != gen)
                {
                    mark.emitted_gen = gen;
                    model.template update_feature<false, perspective>(idx);
                }
            }
            const int n_new = new_idx.size();
            for (int i = 0; i < n_new; ++i)
            {
                if (i + PrefetchDistance < n_new)
                    model.prefetch_feature(new_idx[i + PrefetchDistance]);

                const int idx = new_idx[i];
                ThreatMark &mark = threat_marks[idx];
                if (mark.old_gen != gen && mark.emitted_gen != gen)
                {
                    mark.emitted_gen = gen;
                    model.template update_feature<true, perspective>(idx);
                }
            }
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

        // Reads `count` flat scalar values of type T (int8/int16/int32)
        // directly into `dst` (of type U, possibly wider -- e.g. widening
        // int8 into int16 storage), transparently handling the
        // "COMPRESSED_LEB128" marker. `dst` must have room for `count`
        // elements; the caller owns its storage (array, vector, etc.).
        //
        // When T == U, this reads straight into `dst` with no intermediate
        // buffer at all. When widening is needed, the temporary raw-T buffer
        // is allocated for-overwrite (no zero-init) since every element is
        // immediately overwritten by either file.read() or the cast loop
        // below -- a prior version used std::vector for both `dst`'s backing
        // storage and this temporary, paying for a zero-init memset on both
        // that was wholly wasted (profiled to ~20% of NNUE-loading CPU time).
        template <typename T, typename U>
        static void read_tensor_flat_into(std::ifstream &file, U *dst, std::size_t count, const std::string &label)
        {
            if (!next_is_leb128_marker(file))
            {
                if constexpr (std::is_same_v<T, U>)
                {
                    file.read(reinterpret_cast<char *>(dst), static_cast<std::streamsize>(count * sizeof(T)));
                    check(file, label);
                }
                else
                {
                    auto raw = std::make_unique_for_overwrite<T[]>(count);
                    file.read(reinterpret_cast<char *>(raw.get()), static_cast<std::streamsize>(count * sizeof(T)));
                    check(file, label);
                    for (std::size_t i = 0; i < count; ++i)
                        dst[i] = static_cast<U>(raw[i]);
                }
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
                dst[i] = static_cast<U>(decoded[i]);
        }

        template <typename Layer, int In, int Out>
        static Layer read_dense_layer(std::ifstream &file, const std::string &name)
        {
            // Uninitialized: read_tensor_flat_into() below immediately fills
            // every element, so value-initializing here would just be a
            // wasted memset (weights[0].data() is contiguous over the full
            // In*Out range since std::array packs its elements with no
            // padding, matching the file's flat row-major layout).
            std::array<std::int32_t, Out> biases;
            std::array<std::array<std::int8_t, In>, Out> weights;

            read_tensor_flat_into<std::int32_t>(file, biases.data(), Out, name + " biases");
            read_tensor_flat_into<std::int8_t>(file, weights.data()->data(), static_cast<std::size_t>(In) * Out, name + " weights");

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
            auto accumulator_biases_ptr = std::make_unique_for_overwrite<std::array<std::int16_t, L1>>();
            read_tensor_flat_into<std::int16_t>(file, accumulator_biases_ptr->data(), L1, "accumulator biases");

            // Feature transformer weights, written as two independent segments
            // (Full_Threats then HalfKAv2_hm^, matching the "Full_Threats+HalfKAv2_hm^"
            // feature-name split order): each segment has its own weight tensor
            // (int8 for Full_Threats, int16 for HalfKAv2_hm^) and its own
            // int32 PSQT tensor.
            // for-overwrite: both arrays (~170MB + ~2.7MB) are fully
            // populated row-by-row by read_segment() right below: an initial
            // value-init here would zero all of it for nothing.
            auto accumulator_weights_ptr = std::make_unique_for_overwrite<std::array<std::array<std::int16_t, L1>, NumFeatures>>();
            auto psqt_weights_ptr = std::make_unique_for_overwrite<std::array<std::array<std::int32_t, NumPsqtBuckets>, NumFeatures>>();
            auto &accumulator_weights = *accumulator_weights_ptr;
            auto &psqt_weights = *psqt_weights_ptr;

            // Reads straight into accumulator_weights[row_offset..]/
            // psqt_weights[row_offset..]: rows within a segment are
            // contiguous (std::array packs elements with no padding), and
            // the file stores each segment in matching row-major order, so
            // no intermediate flat buffer + row-copy loop is needed.
            auto read_segment = [&](int row_offset, int n_rows, bool weight_is_int8)
            {
                std::int16_t *weight_dst = accumulator_weights[row_offset].data();
                if (weight_is_int8)
                    read_tensor_flat_into<std::int8_t>(file, weight_dst, static_cast<std::size_t>(n_rows) * L1, "ft weight (int8 segment)");
                else
                    read_tensor_flat_into<std::int16_t>(file, weight_dst, static_cast<std::size_t>(n_rows) * L1, "ft weight (int16 segment)");

                std::int32_t *psqt_dst = psqt_weights[row_offset].data();
                read_tensor_flat_into<std::int32_t>(file, psqt_dst, static_cast<std::size_t>(n_rows) * NumPsqtBuckets, "ft psqt weight");
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
