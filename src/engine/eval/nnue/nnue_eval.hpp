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
// VBoard uses it only on load. Moves are handled incrementally and *lazily*:
// VBoard::play buffers each ply's feature changes into a PendingDiff
// (push_halfka_piece() per moved/captured piece, collect_threats_scoped()/
// filter_threats_diff() for the Full_Threats scoped recompute -- see
// full_threats_incremental.hpp -- and collect_full_perspective() for the
// mover's perspective on king moves), and the accumulator is only updated
// by materialize() when an eval is actually requested. See the design note
// on NnueEval::model and virtual_board.hpp.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common/logger.hpp"
#include "common/fatal.hpp"
#include "common/cpu.hpp"
#include "core/board/board.hpp"
#include "engine/config/nnue.hpp"
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
    // (initialize_perspective/apply_list): a single iteration of
    // lookahead doesn't give the ~500-cycle DRAM round trip (measured on
    // this machine) enough time to complete before update_feature() actually
    // reads that row, since one update_feature() call only takes ~100-120
    // cycles with warm data. Tuned empirically via perf record self-time on
    // the eager apply pass (bench 12, single core): 4=31.65%, 6=28.11%,
    // 8=28.26%, 12=28.23% -- 6 captures essentially all of the available
    // gain, higher distances don't help further.
    constexpr int PrefetchDistance = 6;

    // Capacity of each per-ply pending add/remove list (see PendingDiff /
    // NnueEval's lazy-apply design below). Bounds, per perspective per ply:
    //   - non-king move: filtered scoped-threat diff (<= ~981 unique entries,
    //     see MAX_THREAT_FEATURES's derivation) + <= 3 HalfKA piece toggles;
    //   - king move (mover's perspective): full-board activation list,
    //     <= MAX_FULL_SCAN_THREAT_FEATURES(512) threats + <= 32 pieces.
    // FixedIntList FATALs (rather than truncating) if these bounds are ever
    // exceeded, so this can't silently desync the accumulator.
    constexpr int MAX_PENDING_FEATURES = 1024;
    static_assert(MAX_PENDING_FEATURES >= 981 + 8, "must cover a filtered scoped diff plus HalfKA toggles");
    static_assert(MAX_PENDING_FEATURES >= threats::MAX_FULL_SCAN_THREAT_FEATURES + 64, "must cover a king-move full-perspective activation list");

    // One ply's worth of buffered accumulator work (see NnueEval's lazy-apply
    // design): per perspective, the feature indices to deactivate/activate,
    // plus a `refresh` flag for the king-move case where that perspective is
    // instead rebuilt from scratch (reset + `add` list holds the full
    // post-move activation set; `remove` is unused).
    struct PendingDiff
    {
        threats::FixedIntList<MAX_PENDING_FEATURES> remove[2];
        threats::FixedIntList<MAX_PENDING_FEATURES> add[2];
        bool refresh[2];

        void clear()
        {
            remove[WHITE].clear();
            remove[BLACK].clear();
            add[WHITE].clear();
            add[BLACK].clear();
            refresh[WHITE] = refresh[BLACK] = false;
        }
    };

    class NnueEval
    {
    public:
        using Model = NnueModel<NumFeatures, NumFullThreatsFeatures, L1, NumPsqtBuckets, NumLsBuckets, L2, L3>;

    private:
        // Lazy apply: play() no longer touches the accumulator at all.
        // Instead VBoard::play fills a PendingDiff (via begin_pending()/
        // commit_pending()) with the feature indices the move changes --
        // *collection* stays eager, since it needs the board exactly as it is
        // around Board::play(), but the memory-bandwidth-bound part (random
        // weight-row reads in update_feature, plus the 4.2KB push_state
        // snapshot) is deferred to materialize(), which runs only when an
        // eval is actually requested. Measured (bench 10): only ~268k
        // get_result calls for ~565k plays -- nearly half of all played nodes
        // are cut off (TT hits, etc.) without ever evaluating, so their
        // buffered diffs get discarded by unplay_pop() for free.
        //
        // Invariants: applied_depth <= lazy_depth and psqt_applied_depth <=
        // lazy_depth (the two are otherwise unordered relative to each
        // other); the L1 accumulator holds the position at ply
        // `applied_depth` and the PSQT accumulator the one at ply
        // `psqt_applied_depth`; the model's two snapshot stacks are exactly
        // `applied_depth` / `psqt_applied_depth` deep (one snapshot pushed
        // per materialized ply, so unplaying a materialized ply is a
        // pop-state memcpy, and unplaying a never-evaluated ply is just a
        // counter decrement). The PSQT half materializes on its own (see
        // materialize_psqt / evaluate_psqt_abs): pruning heuristics ask for
        // the cheap PSQT estimate far more often than a full eval happens,
        // and its rows are 32B instead of 2KB.
        //
        // `model` and the lazy counters are mutable because materialization
        // is triggered by evaluate_abs()/evaluate_psqt_abs(), which are
        // logically const (callers hold const VBoard&s); materializing never
        // changes any observable eval result, it only catches the cached
        // accumulator state up.
        mutable Model model;
        std::unique_ptr<PendingDiff[]> pending = std::make_unique_for_overwrite<PendingDiff[]>(constants::MaxHistorySize);
        mutable int lazy_depth = 0;
        mutable int applied_depth = 0;
        mutable int psqt_applied_depth = 0;

        // Depth tag for each L1 snapshot on the model's stack: the value
        // applied_depth must return to when that snapshot is popped. The
        // per-ply replay path pushes one snapshot per ply (tag ==
        // applied_depth just before the ply is applied, so popping is the
        // old `--applied_depth`), but the from-scratch rebuild path (see
        // materialize(board)) jumps several plies with a SINGLE snapshot
        // whose tag is the pre-jump applied_depth: popping it lands back
        // below every skipped ply, whose pending diffs are still buffered
        // and can re-materialize later if needed. Stays in lockstep with the
        // model's acc snapshot stack (same pushes/pops). The PSQT stack
        // keeps the strict one-snapshot-per-ply discipline and needs no
        // tags.
        std::unique_ptr<int[]> acc_snapshot_base = std::make_unique_for_overwrite<int[]>(constants::MaxHistorySize);
        mutable int acc_snapshot_count = 0;

        // Reusable, allocated-once membership-stamp buffer for
        // filter_threats_diff()'s dedup/diff (see below): sized to
        // Full_Threats' NUM_INPUTS so every raw feature index collected by
        // the scoped-recompute path (full_threats_incremental.hpp) can be
        // used directly as an index into it. Zero-initialized once here
        // (construction/load time, not per search node); filter_threats_diff
        // never clears it, it just bumps diff_generation so every stamp
        // from prior calls compares unequal to the current one.
        //
        // The three stamps for a given idx (old/new/emitted) are packed into
        // one struct (array-of-structs) rather than three separate
        // NUM_INPUTS-sized arrays (structure-of-arrays): filter_threats_diff
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

        // materialize() helper for the copy operations below: catching the
        // source up (both halves) before copying means the copy only has to
        // duplicate the model (whose snapshot stacks then exactly match
        // lazy_depth) and the counters -- never the pending buffers
        // themselves.
        const Model &materialized_model() const
        {
            materialize();
            materialize_psqt();
            return model;
        }

    public:
        NnueEval(Model &&_model) : model(std::move(_model)) {}

        // Copies from a process-wide cached prototype (see
        // shared_base_model) instead of re-reading the ~170MB file: every
        // layer's weight tables are shared_ptr<const>-shared, so this copy
        // is just fresh accumulator state plus refcount bumps. Constructing
        // VBoards used to reload the file from disk every time (~measurable
        // fraction of a whole bench run spent in sys time).
        explicit NnueEval(const std::string &path)
            : model(shared_base_model(path))
        {
        }

        // Declared explicitly: `pending` is a unique_ptr<T[]>, which isn't
        // copyable. Copies only happen at thread/VBoard setup (see VBoard's
        // copy paths), so the strategy is to materialize the source first
        // (see materialized_model()) rather than deep-copy pending buffers:
        // afterwards applied_depth == lazy_depth and there is nothing
        // pending to copy. The fresh `pending` array from the NSDMI is left
        // as scratch space.
        NnueEval(const NnueEval &other)
            : model(other.materialized_model()),
              lazy_depth(other.lazy_depth),
              applied_depth(other.lazy_depth),
              psqt_applied_depth(other.lazy_depth),
              acc_snapshot_count(other.acc_snapshot_count),
              threat_marks(other.threat_marks),
              diff_generation(other.diff_generation)
        {
            std::copy_n(other.acc_snapshot_base.get(), acc_snapshot_count, acc_snapshot_base.get());
        }

        NnueEval &operator=(const NnueEval &other)
        {
            if (this != &other)
            {
                model = other.materialized_model();
                threat_marks = other.threat_marks;
                diff_generation = other.diff_generation;
                lazy_depth = other.lazy_depth;
                applied_depth = other.lazy_depth;
                psqt_applied_depth = other.lazy_depth;
                acc_snapshot_count = other.acc_snapshot_count;
                std::copy_n(other.acc_snapshot_base.get(), acc_snapshot_count, acc_snapshot_base.get());
            }
            return *this;
        }

        NnueEval(NnueEval &&) noexcept = default;
        NnueEval &operator=(NnueEval &&) noexcept = default;

        // Which accumulator halves a full perspective rescan rebuilds: Both
        // for initialize() (fresh baseline for everything), AccOnly for the
        // lazy from-scratch catch-up in materialize(board) -- there the PSQT
        // half has its own independent counter/snapshot stack and MUST NOT
        // be touched out of band.
        enum class RebuildScope
        {
            Both,
            AccOnly
        };

        // Full recompute of a single perspective's accumulator contribution
        // (Full_Threats + HalfKA), scanning the whole board -- used by
        // initialize() below (both perspectives), and by materialize(board)
        // as the from-scratch alternative to replaying a long backlog of
        // buffered per-ply diffs.
        template <Color perspective, RebuildScope scope = RebuildScope::Both>
        void initialize_perspective(const Board &board) const
        {
            if constexpr (scope == RebuildScope::Both)
                model.template reset_perspective<perspective>();
            else
                model.template reset_acc_perspective<perspective>();

            const auto prefetch = [this](int idx)
            {
                if constexpr (scope == RebuildScope::Both)
                    model.prefetch_feature(idx);
                else
                    model.prefetch_acc_feature(idx);
            };
            const auto activate = [this](int idx)
            {
                if constexpr (scope == RebuildScope::Both)
                    model.template update_feature<true, perspective>(idx);
                else
                    model.template update_acc_feature<true, perspective>(idx);
            };

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
                        prefetch(idx);
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
                    prefetch(perspective_threats[lookahead]);
                else if (const int piece_idx = lookahead - n_threats; piece_idx < n_pieces)
                    prefetch(piece_features[piece_idx]);
                activate(perspective_threats[i]);
            }

            for (int i = 0; i < n_pieces; ++i)
            {
                // Same spill idea in reverse doesn't apply here (nothing comes
                // after piece_features), but the threats loop above already
                // primed the first PrefetchDistance piece entries by the time
                // we get here, so this loop's own cold-start is the piece
                // count it spilled into, not PrefetchDistance from zero.
                if (i + PrefetchDistance < n_pieces)
                    prefetch(piece_features[i + PrefetchDistance]);
                activate(piece_features[i]);
            }
        }

        // Full recompute (see scope note above): rebuilds both perspectives'
        // accumulators from scratch by scanning the whole board. Also drops
        // all lazy/snapshot state -- the rebuilt accumulator IS the new
        // baseline, anything buffered against the old one is meaningless.
        void initialize(const Board &board)
        {
            lazy_depth = 0;
            applied_depth = 0;
            psqt_applied_depth = 0;
            acc_snapshot_count = 0;
            model.reset_snapshot_stack();
            initialize_perspective<WHITE>(board);
            initialize_perspective<BLACK>(board);
        }

        // `board` must be the position at the current ply (lazy_depth) --
        // it's both the eval target and, when the buffered backlog is long
        // enough, the source for a from-scratch accumulator rebuild (see
        // materialize(board)).
        std::int32_t evaluate_abs(const Board &board) const
        {
            const int piece_count = std::popcount(board.get_occupancy<NO_COLOR>());
            // get_result() reads both the L1 accumulator and the PSQT
            // buckets, so both halves must be caught up.
            materialize(board);
            materialize_psqt();
            return board.get_side_to_move() == WHITE
                       ? model.template get_result<WHITE>(piece_count)
                       : model.template get_result<BLACK>(piece_count);
        }

        // PSQT-only fast eval (see NnueModel::get_psqt_result): the trained
        // material/PSQT estimate, used by pruning heuristics in place of the
        // HCE EvalState estimate.
        std::int32_t evaluate_psqt_abs(Color side_to_move, int piece_count) const
        {
            materialize_psqt();
            return side_to_move == WHITE
                       ? model.template get_psqt_result<WHITE>(piece_count)
                       : model.template get_psqt_result<BLACK>(piece_count);
        }

        // Incremental HalfKAv2_hm^ update for a single piece (add or remove),
        // analogous to v1's per-piece feature update, buffered into `pd`
        // instead of applied (see the lazy-apply design note above). Not
        // valid for the piece whose own move is a king move -- callers must
        // set pd.refresh for that perspective instead (see
        // collect_full_perspective), since a king move changes every
        // HalfKAv2_hm^ feature for that perspective (king square/bucket is
        // baked into every other piece's index).
        template <bool activate, Color perspective>
        void push_halfka_piece(PendingDiff &pd, int king_sq, Color piece_color, Piece piece_type, int piece_sq) const
        {
            if (piece_type == NO_PIECE)
                return;
            const int idx = NumFullThreatsFeatures + halfka::feature_index<perspective>(king_sq, piece_color, piece_type, piece_sq);
            (activate ? pd.add : pd.remove)[perspective].push_back(idx);
        }

        // Collection-only counterpart of initialize_perspective() for the
        // king-move lazy path: gathers the perspective's complete post-move
        // activation set (HalfKA piece features + Full_Threats full scan)
        // into `out` without touching the accumulator. Applied later by
        // materialize() as reset_perspective + activate-all (pd.refresh).
        template <Color perspective>
        void collect_full_perspective(const Board &board, threats::FixedIntList<MAX_PENDING_FEATURES> &out) const
        {
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
                        out.push_back(NumFullThreatsFeatures + halfka::feature_index<perspective>(ksq, color, piece_type, sq));
                    }
                }
            }

            threats::FixedIntList<threats::MAX_FULL_SCAN_THREAT_FEATURES> perspective_threats;
            threats::fill_features<perspective>(board, perspective_threats);
            for (const int idx : perspective_threats)
                out.push_back(idx);
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
        // move, one from "after" -- and appends the resulting add/remove set
        // to `out_remove`/`out_add` (a PendingDiff's lists, which already
        // hold the move's HalfKA toggles). Whichever list is passed as
        // `old_idx` ends up removed and whichever is passed as `new_idx`
        // ends up added when the diff is materialized.
        //
        // Zero heap allocation, zero std::sort/std::unique: membership in
        // each list is recorded via a per-call "generation" stamp in reusable
        // NUM_INPUTS-sized member arrays (allocated once, at construction --
        // not a per-call/per-search-node allocation), so both dedup and the
        // add/remove diff are done in O(old_idx.size() + new_idx.size())
        // without ever clearing the arrays themselves.
        //
        // Filtering here (at collect time) rather than at apply time keeps
        // the pending buffers small and, more importantly, keeps the
        // measured ~50% of collected entries that are unchanged features
        // (present in both lists) from ever being buffered, prefetched or
        // applied.
        void filter_threats_diff(
            const threats::FixedIntList<threats::MAX_THREAT_FEATURES> &old_idx,
            const threats::FixedIntList<threats::MAX_THREAT_FEATURES> &new_idx,
            threats::FixedIntList<MAX_PENDING_FEATURES> &out_remove,
            threats::FixedIntList<MAX_PENDING_FEATURES> &out_add)
        {
            const std::uint32_t gen = ++diff_generation;

            for (int idx : old_idx)
                threat_marks[idx].old_gen = gen;
            for (int idx : new_idx)
                threat_marks[idx].new_gen = gen;

            for (int idx : old_idx)
            {
                ThreatMark &mark = threat_marks[idx];
                if (mark.new_gen != gen && mark.emitted_gen != gen)
                {
                    mark.emitted_gen = gen;
                    out_remove.push_back(idx);
                }
            }
            for (int idx : new_idx)
            {
                ThreatMark &mark = threat_marks[idx];
                if (mark.old_gen != gen && mark.emitted_gen != gen)
                {
                    mark.emitted_gen = gen;
                    out_add.push_back(idx);
                }
            }
        }

        // ---- Lazy-apply ply bookkeeping (see the design note on `model`) --
        // VBoard::play fills the returned PendingDiff, then commits it; the
        // accumulator itself is only touched by materialize() below.
        PendingDiff &begin_pending()
        {
            PendingDiff &pd = pending[lazy_depth];
            pd.clear();
            return pd;
        }

        void commit_pending()
        {
            ++lazy_depth;
        }

        // Undoes one play(): each half that was materialized at this ply (an
        // eval of its kind happened at or below it) must be rolled back via
        // its snapshot; a half that never caught up simply abandons its part
        // of the buffered diff -- the whole point of deferring the apply.
        void unplay_pop()
        {
            if (applied_depth == lazy_depth)
            {
                model.pop_acc_state();
                // Not necessarily lazy_depth - 1: a from-scratch rebuild
                // (materialize(board)) covers several plies with one
                // snapshot, so popping it rewinds applied_depth below every
                // ply that snapshot skipped (their diffs are still buffered).
                applied_depth = acc_snapshot_base[--acc_snapshot_count];
            }
            if (psqt_applied_depth == lazy_depth)
            {
                model.pop_psqt_state();
                --psqt_applied_depth;
            }
            --lazy_depth;
        }

        // Catches the L1 accumulator up to the current ply by replaying
        // every still-pending buffered diff, snapshotting before each one so
        // unplay_pop() can roll back through materialized plies. The PSQT
        // half is NOT touched here -- it has its own pass below.
        void materialize() const
        {
            while (applied_depth < lazy_depth)
            {
                model.push_acc_state();
                acc_snapshot_base[acc_snapshot_count++] = applied_depth;
                const PendingDiff &pd = pending[applied_depth];
                apply_pending_acc<WHITE>(pd);
                apply_pending_acc<BLACK>(pd);
                ++applied_depth;
            }
        }

        // Board-aware variant used by evaluate_abs: when the backlog is long
        // enough that replaying it ply by ply costs more than a full
        // two-perspective rescan of the current position (threshold measured
        // empirically, lower when a buffered king move already forces a
        // one-perspective reset+full replay -- see engine_constants::nnue),
        // rebuild the L1 accumulator from scratch off `board` under a single
        // depth-tagged snapshot instead. Skipped plies stay buffered: after
        // unwinding back through the jump, they can still re-materialize.
        void materialize([[maybe_unused]] const Board &board) const
        {
            const int backlog = lazy_depth - applied_depth;
            if (backlog < engine_constants::nnue::AccRebuildKingMinPlies)
            {
                materialize();
                return;
            }

            if (backlog < engine_constants::nnue::AccRebuildMinPlies)
            {
                bool has_refresh = false;
                for (int d = applied_depth; d < lazy_depth && !has_refresh; ++d)
                    has_refresh = pending[d].refresh[WHITE] || pending[d].refresh[BLACK];
                if (!has_refresh)
                {
                    materialize();
                    return;
                }
            }

            model.push_acc_state();
            acc_snapshot_base[acc_snapshot_count++] = applied_depth;
#ifdef NNUE_REBUILD_RESCAN // full-board rescan variant (measured slower, kept for A/B)
            initialize_perspective<WHITE, RebuildScope::AccOnly>(board);
            initialize_perspective<BLACK, RebuildScope::AccOnly>(board);
#else
            apply_backlog_batched<WHITE>();
            apply_backlog_batched<BLACK>();
#endif
            applied_depth = lazy_depth;
        }

        // Same catch-up for the PSQT half alone: 32B rows instead of 2KB,
        // 64B snapshots instead of ~4KB, so a pruning heuristic can get the
        // trained material/PSQT estimate at plies where no full eval ever
        // happens without paying for the L1 half.
        void materialize_psqt() const
        {
            while (psqt_applied_depth < lazy_depth)
            {
                model.push_psqt_state();
                const PendingDiff &pd = pending[psqt_applied_depth];
                apply_pending_psqt<WHITE>(pd);
                apply_pending_psqt<BLACK>(pd);
                ++psqt_applied_depth;
            }
        }

#ifdef CHESS26_UNIT_TESTING
        const auto &get_accumulator() const
        {
            materialize();
            return model.get_accumulator();
        }
#endif

    private:
        // materialize()'s apply pass for one buffered list (L1 half), with
        // the same software-prefetch lookahead the eager path used (see
        // PrefetchDistance): every entry here survived filtering, so every
        // prefetch corresponds to a weight row that's really about to be
        // read.
        template <bool activate, Color perspective>
        void apply_acc_list(const threats::FixedIntList<MAX_PENDING_FEATURES> &list) const
        {
            const int n = list.size();
            for (int i = 0; i < n; ++i)
            {
                if (i + PrefetchDistance < n)
                    model.prefetch_acc_feature(list[i + PrefetchDistance]);
                model.template update_acc_feature<activate, perspective>(list[i]);
            }
        }

        // PSQT-half twin of apply_acc_list. Prefetch kept even though rows
        // are tiny (32B): the PSQT weight table is 2.7MB, so rows still miss
        // in L1/L2 routinely.
        template <bool activate, Color perspective>
        void apply_psqt_list(const threats::FixedIntList<MAX_PENDING_FEATURES> &list) const
        {
            const int n = list.size();
            for (int i = 0; i < n; ++i)
            {
                if (i + PrefetchDistance < n)
                    model.prefetch_psqt_feature(list[i + PrefetchDistance]);
                model.template update_psqt_feature<activate, perspective>(list[i]);
            }
        }

        // Batched replay of the whole [applied_depth, lazy_depth) backlog
        // for one perspective, without intermediate snapshots: a king move
        // by `perspective` anywhere in the range makes every earlier ply's
        // work for that perspective dead (its refresh resets the perspective
        // and its add list already holds the full post-move activation set,
        // collected at play time -- no board rescan needed), so replay
        // starts at the LAST refresh ply and skips everything before it.
        template <Color perspective>
        void apply_backlog_batched() const
        {
            int start = applied_depth;
            for (int d = lazy_depth - 1; d >= applied_depth; --d)
                if (pending[d].refresh[perspective])
                {
                    start = d;
                    break;
                }
            for (int d = start; d < lazy_depth; ++d)
                apply_pending_acc<perspective>(pending[d]);
        }

        template <Color perspective>
        void apply_pending_acc(const PendingDiff &pd) const
        {
            // refresh (king move by `perspective`): the add list holds the
            // full post-move activation set over a reset accumulator, and
            // the remove list is unused by construction (VBoard pushes no
            // per-piece toggles for the refreshed perspective).
            if (pd.refresh[perspective])
                model.template reset_acc_perspective<perspective>();
            else
                apply_acc_list<false, perspective>(pd.remove[perspective]);
            apply_acc_list<true, perspective>(pd.add[perspective]);
        }

        template <Color perspective>
        void apply_pending_psqt(const PendingDiff &pd) const
        {
            if (pd.refresh[perspective])
                model.template reset_psqt_perspective<perspective>();
            else
                apply_psqt_list<false, perspective>(pd.remove[perspective]);
            apply_psqt_list<true, perspective>(pd.add[perspective]);
        }

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

        // Process-wide model cache: the file is read and decoded once per
        // distinct path, then every NnueEval copy-constructs from the cached
        // prototype (cheap: weight tables are shared_ptr<const>-shared
        // across copies, only per-instance accumulator state is duplicated).
        // The prototype itself stays alive for the program's lifetime -- it
        // IS the weight storage every instance points into.
        //
        // Thread-safety: the mutex serializes lookups/inserts; the returned
        // reference stays valid outside the lock because unordered_map is
        // node-based (inserts never invalidate references to existing
        // elements) and cached entries are never mutated or erased.
        static const Model &shared_base_model(const std::string &path)
        {
            static std::mutex cache_mutex;
            static std::unordered_map<std::string, Model> cache;

            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(path);
            if (it == cache.end())
                it = cache.emplace(path, load_model(path)).first;
            return it->second;
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
            // and its own int32 PSQT tensor. Full_Threats' weight tensor is
            // kept int8 in memory (not widened to int16 like before) -- see
            // AccumulatorLayer's class comment for why: it halves the bytes
            // AccumulatorLayer::apply_row has to fetch from DRAM per update
            // on the hot incremental-update path, at the cost of a cheap
            // sign-extend done at apply time instead of load time.
            // for-overwrite: all arrays are fully populated row-by-row by
            // read_segment()/below: an initial value-init here would zero
            // all of it for nothing.
            auto threat_weights_ptr = std::make_unique_for_overwrite<std::array<std::array<std::int8_t, L1>, NumFullThreatsFeatures>>();
            auto halfka_weights_ptr = std::make_unique_for_overwrite<std::array<std::array<std::int16_t, L1>, NumHalfkaFeatures>>();
            auto psqt_weights_ptr = std::make_unique_for_overwrite<std::array<std::array<std::int32_t, NumPsqtBuckets>, NumFeatures>>();
            auto &threat_weights = *threat_weights_ptr;
            auto &halfka_weights = *halfka_weights_ptr;
            auto &psqt_weights = *psqt_weights_ptr;

            // Full_Threats segment: weight tensor read directly as int8 (no
            // widening -- the file already stores it that way).
            read_tensor_flat_into<std::int8_t>(file, threat_weights.data()->data(), static_cast<std::size_t>(NumFullThreatsFeatures) * L1, "ft weight (int8 segment)");
            read_tensor_flat_into<std::int32_t>(file, psqt_weights[0].data(), static_cast<std::size_t>(NumFullThreatsFeatures) * NumPsqtBuckets, "ft psqt weight (threats)");

            // HalfKAv2_hm^ segment: weight tensor stays int16, at row offset
            // NumFullThreatsFeatures within the combined PSQT table (its own
            // weight table is separate/zero-based, matching AccumulatorLayer's
            // halfka_weights indexing of feature - NumFullThreatsFeatures).
            read_tensor_flat_into<std::int16_t>(file, halfka_weights.data()->data(), static_cast<std::size_t>(NumHalfkaFeatures) * L1, "ft weight (int16 segment)");
            read_tensor_flat_into<std::int32_t>(file, psqt_weights[NumFullThreatsFeatures].data(), static_cast<std::size_t>(NumHalfkaFeatures) * NumPsqtBuckets, "ft psqt weight (halfka)");

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
                std::move(*threat_weights_ptr),
                std::move(*halfka_weights_ptr),
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
