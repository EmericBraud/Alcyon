#pragma once

// A Board that also maintains evaluation state incrementally across
// play/unplay, so the (lazy and full) evals are almost free to request:
// the NNUE accumulator diffs in NNUE builds, the HCE EvalState (PST/material/
// phase/pawn-key increments) in legacy builds. Each build mode carries only
// its own state -- EvalState does not exist at all under NNUE_EVAL.

#include "core/piece/color.hpp"
#include "core/board/board.hpp"
#include "common/file.hpp"
#ifdef NNUE_EVAL
#include "engine/eval/nnue/nnue_eval.hpp"
#include "engine/eval/nnue/fixed_int_list.hpp"
#else
#include "engine/eval/hce/move_eval_increment.hpp"
#endif

class VBoard : public Board
{
#ifdef NNUE_EVAL
#define NNUE_FULL_MODEL_PATH file::get_data_path("nnue/v2.nnue")
    nnue::NnueEval nnue_eval{NNUE_FULL_MODEL_PATH};

    template <bool activate>
    inline void push_nnue_halfka_both_perspectives(nnue::PendingDiff &pd, int white_king_sq, int black_king_sq, Color piece_color, Piece piece_type, int piece_sq)
    {
        nnue_eval.template push_halfka_piece<activate, WHITE>(pd, white_king_sq, piece_color, piece_type, piece_sq);
        nnue_eval.template push_halfka_piece<activate, BLACK>(pd, black_king_sq, piece_color, piece_type, piece_sq);
    }
#else
    // Legacy HCE fast-eval state, incremented/decremented on every
    // play/unplay and read by Eval::lazy_eval_relative. NNUE builds don't
    // carry it at all -- the network's PSQT head plays that role there (see
    // NnueEval::evaluate_psqt_abs).
    EvalState eval_state;
#endif
public:
    VBoard &operator=(const VBoard &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

#ifdef NNUE_EVAL
            nnue_eval = other.nnue_eval;
#else
            eval_state = other.eval_state;
#endif
        };

        return *this;
    }
    VBoard &operator=(const Board &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

#ifdef NNUE_EVAL
            // Reuse the member's already-loaded model: initialize() rebuilds
            // the accumulator from `other`'s position and drops all
            // lazy/snapshot state, which is everything a fresh NnueEval
            // would have provided.
            nnue_eval.initialize(other);
#else
            eval_state = EvalState(pieces_occ);
#endif
        };

        return *this;
    }
    VBoard(const VBoard &other)
        : Board(other)
#ifdef NNUE_EVAL
          ,
          nnue_eval(other.nnue_eval)
#else
          ,
          eval_state(other.eval_state)
#endif
    {
    }
    VBoard(const Board &other)
        : Board(other)
#ifdef NNUE_EVAL
          ,
          nnue_eval{NNUE_FULL_MODEL_PATH}
#else
          ,
          eval_state(other.get_all_bitboards())
#endif
    {
#ifdef NNUE_EVAL
        nnue_eval.initialize(other);
#endif
    }

    VBoard(VBoard &&other) noexcept
        : Board(std::move(other))
#ifdef NNUE_EVAL
          ,
          nnue_eval(std::move(other.nnue_eval))
#else
          ,
          eval_state(std::move(other.eval_state))
#endif
    {
    }

    VBoard &operator=(VBoard &&other) noexcept
    {
        if (this != &other)
        {
            Board::operator=(std::move(other));
#ifdef NNUE_EVAL
            nnue_eval = std::move(other.nnue_eval);
#else
            eval_state = std::move(other.eval_state);
#endif
        }
        return *this;
    }

    VBoard() : Board()
    {
    }

    bool load_fen(const std::string_view fen_string)
    {
        bool r = Board::load_fen(fen_string);
#ifdef NNUE_EVAL
        nnue_eval.initialize(*this);
#else
        eval_state = EvalState(get_all_bitboards());
#endif

        return r;
    }

    inline void play(const Move move)
    {
        if (get_side_to_move() == WHITE)
            play<WHITE>(move);
        else
            play<BLACK>(move);
    }
    inline void unplay(const Move move)
    {
        if (get_side_to_move() == WHITE)
            unplay<BLACK>(move);
        else
            unplay<WHITE>(move);
    }

#ifdef NNUE_EVAL
    // King moves invalidate every HalfKA index and the Full_Threats
    // orientation for the MOVER's own perspective only (both are keyed on
    // that perspective's own king square) -- the other perspective's own
    // king hasn't moved, so it can be updated incrementally exactly like any
    // other piece move, restricted to that one perspective. Splitting the
    // work this way avoids a full two-perspective rescan (NnueEval::
    // initialize()) on every king move, which used to redo the entire
    // non-mover perspective's work for nothing.
    template <Color Us>
    inline void play_king_move(const Move move)
    {
        // Lazy apply (see NnueEval's design note): nothing here touches the
        // accumulator. The mover's perspective is flagged for a full
        // deferred refresh (reset + complete post-move activation set), and
        // Them's incremental diff is buffered -- all of it materialized only
        // if an eval actually happens at or below this ply.
        nnue::PendingDiff &pd = nnue_eval.begin_pending();
        pd.refresh[Us] = true;

        constexpr Color Them = static_cast<Color>(!Us);
        const int them_king_sq = king_sq[Them];
        const int from_sq = move.get_from_sq();
        const int to_sq = move.get_to_sq();
        const uint32_t flags = move.get_flags();
        const Piece to_piece = move.get_to_piece();

        nnue::threats::FixedIntList<nnue::threats::MAX_TOUCHED_SQUARES> touched_squares;
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> old_them_threats;

        touched_squares.push_back(from_sq);
        touched_squares.push_back(to_sq);

        if (to_piece != NO_PIECE)
            nnue_eval.template push_halfka_piece<false, Them>(pd, them_king_sq, Them, to_piece, to_sq);

        int rook_from = -1, rook_to = -1;
        if (flags == Move::Flags::KING_CASTLE)
        {
            rook_from = (Us == WHITE) ? Square::h1 : Square::h8;
            rook_to = (Us == WHITE) ? Square::f1 : Square::f8;
        }
        else if (flags == Move::Flags::QUEEN_CASTLE)
        {
            rook_from = (Us == WHITE) ? Square::a1 : Square::a8;
            rook_to = (Us == WHITE) ? Square::d1 : Square::d8;
        }
        if (rook_from >= 0)
        {
            nnue_eval.template push_halfka_piece<false, Them>(pd, them_king_sq, Us, ROOK, rook_from);
            nnue_eval.template push_halfka_piece<true, Them>(pd, them_king_sq, Us, ROOK, rook_to);
            touched_squares.push_back(rook_from);
            touched_squares.push_back(rook_to);
        }

        // The moving king is itself a tracked HalfKA feature from Them's
        // perspective (the enemy king's square is a real, non-anchor
        // feature -- see halfka_v2_hm_encoder.hpp's header note on the
        // merged own/enemy king plane), unlike from Us's own perspective
        // where the king is the anchor and never toggled as a feature.
        nnue_eval.template push_halfka_piece<false, Them>(pd, them_king_sq, Us, KING, from_sq);
        nnue_eval.template push_halfka_piece<true, Them>(pd, them_king_sq, Us, KING, to_sq);

        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, old_them_threats);

        Board::play<Us>(move);

        nnue_eval.template collect_full_perspective<Us>(*this, pd.add[Us]);

        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> new_them_threats;
        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, new_them_threats);
        nnue_eval.filter_threats_diff(old_them_threats, new_them_threats, pd.remove[Them], pd.add[Them]);

        nnue_eval.commit_pending();
    }
#endif

    template <Color Us>
    inline void play(const Move move)
    {
#ifdef NNUE_EVAL
        const Piece from_piece = move.get_from_piece();
        if (from_piece == KING)
        {
            play_king_move<Us>(move);
            return;
        }

        // Lazy apply (see NnueEval's design note): feature indices are
        // *collected* here, eagerly -- they depend on the board exactly as
        // it stands around Board::play() -- but nothing touches the
        // accumulator; the buffered diff is only materialized if an eval
        // actually happens at or below this ply.
        nnue::PendingDiff &pd = nnue_eval.begin_pending();

        constexpr Color Them = static_cast<Color>(!Us);
        nnue::threats::FixedIntList<nnue::threats::MAX_TOUCHED_SQUARES> touched_squares;
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> old_white_threats, old_black_threats;

        const int white_king_sq = king_sq[WHITE];
        const int black_king_sq = king_sq[BLACK];

        const int from_sq = move.get_from_sq();
        const int to_sq = move.get_to_sq();
        const uint32_t flags = move.get_flags();
        const Piece to_piece = move.get_to_piece();

        push_nnue_halfka_both_perspectives<false>(pd, white_king_sq, black_king_sq, Us, from_piece, from_sq);

        if (flags == Move::Flags::PROMOTION_MASK)
            push_nnue_halfka_both_perspectives<true>(pd, white_king_sq, black_king_sq, Us, move.get_promo_piece(), to_sq);
        else
            push_nnue_halfka_both_perspectives<true>(pd, white_king_sq, black_king_sq, Us, from_piece, to_sq);

        touched_squares.push_back(from_sq);
        touched_squares.push_back(to_sq);

        if (flags == Move::Flags::EN_PASSANT_CAP)
        {
            const int cap_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
            push_nnue_halfka_both_perspectives<false>(pd, white_king_sq, black_king_sq, Them, PAWN, cap_sq);
            touched_squares.push_back(cap_sq);
        }
        else if (to_piece != NO_PIECE)
        {
            push_nnue_halfka_both_perspectives<false>(pd, white_king_sq, black_king_sq, Them, to_piece, to_sq);
        }

        // Zero-copy: collect the pre-move Full_Threats scoped feature set
        // directly from the live (still pre-move) board -- no full Board
        // snapshot needed, since touched_squares only depends on `move`
        // itself. See NnueEval::collect_threats_scoped.
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, old_white_threats, old_black_threats);

        Board::play<Us>(move);

        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> new_white_threats, new_black_threats;
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, new_white_threats, new_black_threats);
        nnue_eval.filter_threats_diff(old_white_threats, new_white_threats, pd.remove[WHITE], pd.add[WHITE]);
        nnue_eval.filter_threats_diff(old_black_threats, new_black_threats, pd.remove[BLACK], pd.add[BLACK]);

        nnue_eval.commit_pending();
#else
        eval_state.increment(move, Us);
        Board::play<Us>(move);
#endif
    }
    template <Color Us>
    inline void unplay(const Move move)
    {
#ifdef NNUE_EVAL
        // Lazy apply makes king and non-king moves symmetric to undo: if
        // this ply's buffered diff was never materialized (no eval happened
        // at or below it), unplay_pop() just abandons it; if it was, the
        // pre-move accumulator comes back via a snapshot memcpy. Either way
        // no collect/diff work is redone here.
        Board::unplay<Us>(move);
        nnue_eval.unplay_pop();
#else
        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);
#endif
    }

#ifndef NNUE_EVAL
    inline EvalState &get_eval_state()
    {
        return eval_state;
    }

    inline const EvalState &get_eval_state() const
    {
        return eval_state;
    }
#endif
#ifdef NNUE_EVAL
    inline auto &get_nnue_eval()
    {
        return nnue_eval;
    }

    inline auto &get_nnue_eval() const
    {
        return nnue_eval;
    }
#endif
};
