#pragma once

// This file aims at creating a board that also implements automatic increment / decrement
// of an Eval State to make the lazy evaluation almost free

#include "core/piece/color.hpp"
#include "core/board/board.hpp"
#include "engine/eval/hce/move_eval_increment.hpp"
#include "common/file.hpp"
#ifdef NNUE_EVAL
#include "engine/eval/nnue/nnue_eval.hpp"
#include "engine/eval/nnue/fixed_int_list.hpp"
#endif

class VBoard : public Board
{
    EvalState eval_state;
#ifdef NNUE_EVAL
#define NNUE_FULL_MODEL_PATH file::get_data_path("nnue/v2.nnue")
    nnue::NnueEval nnue_eval{NNUE_FULL_MODEL_PATH};

    template <bool activate>
    inline void update_nnue_halfka_both_perspectives(int white_king_sq, int black_king_sq, Color piece_color, Piece piece_type, int piece_sq)
    {
        nnue_eval.template update_halfka_piece<activate, WHITE>(white_king_sq, piece_color, piece_type, piece_sq);
        nnue_eval.template update_halfka_piece<activate, BLACK>(black_king_sq, piece_color, piece_type, piece_sq);
    }
#endif
public:
    VBoard &operator=(const VBoard &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

            eval_state = other.eval_state;
#ifdef NNUE_EVAL
            nnue_eval = other.nnue_eval;
#endif
        };

        return *this;
    }
    VBoard &operator=(const Board &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

            eval_state = EvalState(pieces_occ);
#ifdef NNUE_EVAL
            nnue_eval = nnue::NnueEval{NNUE_FULL_MODEL_PATH};
            nnue_eval.initialize(other);
#endif
        };

        return *this;
    }
    VBoard(const VBoard &other)
        : Board(other),
          eval_state(other.eval_state)
#ifdef NNUE_EVAL
          ,
          nnue_eval(other.nnue_eval)
#endif
    {
    }
    VBoard(const Board &other)
        : Board(other),
          eval_state(other.get_all_bitboards())
#ifdef NNUE_EVAL
          ,
          nnue_eval{NNUE_FULL_MODEL_PATH}
#endif
    {
#ifdef NNUE_EVAL
        nnue_eval.initialize(other);
#endif
    }

    VBoard(VBoard &&other) noexcept
        : Board(std::move(other)),
          eval_state(std::move(other.eval_state))
#ifdef NNUE_EVAL
          ,
          nnue_eval(std::move(other.nnue_eval))
#endif
    {
    }

    VBoard &operator=(VBoard &&other) noexcept
    {
        if (this != &other)
        {
            Board::operator=(std::move(other));
            eval_state = std::move(other.eval_state);
#ifdef NNUE_EVAL
            nnue_eval = std::move(other.nnue_eval);
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
        eval_state = EvalState(get_all_bitboards());
#ifdef NNUE_EVAL
        nnue_eval.initialize(*this);
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
            nnue_eval.template update_halfka_piece<false, Them>(them_king_sq, Them, to_piece, to_sq);

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
            nnue_eval.template update_halfka_piece<false, Them>(them_king_sq, Us, ROOK, rook_from);
            nnue_eval.template update_halfka_piece<true, Them>(them_king_sq, Us, ROOK, rook_to);
            touched_squares.push_back(rook_from);
            touched_squares.push_back(rook_to);
        }

        // The moving king is itself a tracked HalfKA feature from Them's
        // perspective (the enemy king's square is a real, non-anchor
        // feature -- see halfka_v2_hm_encoder.hpp's header note on the
        // merged own/enemy king plane), unlike from Us's own perspective
        // where the king is the anchor and never toggled as a feature.
        nnue_eval.template update_halfka_piece<false, Them>(them_king_sq, Us, KING, from_sq);
        nnue_eval.template update_halfka_piece<true, Them>(them_king_sq, Us, KING, to_sq);

        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, old_them_threats);

        eval_state.increment(move, Us);
        Board::play<Us>(move);

        nnue_eval.template initialize_perspective<Us>(*this);

        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> new_them_threats;
        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, new_them_threats);
        nnue_eval.template apply_threats_diff<Them>(old_them_threats, new_them_threats);
    }

    // Symmetric reversal of play_king_move() -- see apply_threats_diff's
    // note on argument order for why (post, pre) is passed on the way back.
    template <Color Us>
    inline void unplay_king_move(const Move move)
    {
        constexpr Color Them = static_cast<Color>(!Us);
        const int them_king_sq = king_sq[Them];
        const int from_sq = move.get_from_sq();
        const int to_sq = move.get_to_sq();
        const uint32_t flags = move.get_flags();
        const Piece to_piece = move.get_to_piece();

        nnue::threats::FixedIntList<nnue::threats::MAX_TOUCHED_SQUARES> touched_squares;
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> post_them_threats;

        touched_squares.push_back(from_sq);
        touched_squares.push_back(to_sq);

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
            touched_squares.push_back(rook_from);
            touched_squares.push_back(rook_to);
        }

        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, post_them_threats);

        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);

        if (to_piece != NO_PIECE)
            nnue_eval.template update_halfka_piece<true, Them>(them_king_sq, Them, to_piece, to_sq);
        if (rook_from >= 0)
        {
            nnue_eval.template update_halfka_piece<true, Them>(them_king_sq, Us, ROOK, rook_from);
            nnue_eval.template update_halfka_piece<false, Them>(them_king_sq, Us, ROOK, rook_to);
        }

        // Reverse the moving king's own HalfKA feature toggle from Them's
        // perspective (see the matching note in play_king_move()).
        nnue_eval.template update_halfka_piece<true, Them>(them_king_sq, Us, KING, from_sq);
        nnue_eval.template update_halfka_piece<false, Them>(them_king_sq, Us, KING, to_sq);

        nnue_eval.template initialize_perspective<Us>(*this);

        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> pre_them_threats;
        nnue_eval.template collect_threats_scoped<Them>(*this, touched_squares, pre_them_threats);
        nnue_eval.template apply_threats_diff<Them>(post_them_threats, pre_them_threats);
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

        constexpr Color Them = static_cast<Color>(!Us);
        nnue::threats::FixedIntList<nnue::threats::MAX_TOUCHED_SQUARES> touched_squares;
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> old_white_threats, old_black_threats;

        const int white_king_sq = king_sq[WHITE];
        const int black_king_sq = king_sq[BLACK];

        const int from_sq = move.get_from_sq();
        const int to_sq = move.get_to_sq();
        const uint32_t flags = move.get_flags();
        const Piece to_piece = move.get_to_piece();

        update_nnue_halfka_both_perspectives<false>(white_king_sq, black_king_sq, Us, from_piece, from_sq);

        if (flags == Move::Flags::PROMOTION_MASK)
            update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Us, move.get_promo_piece(), to_sq);
        else
            update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Us, from_piece, to_sq);

        touched_squares.push_back(from_sq);
        touched_squares.push_back(to_sq);

        if (flags == Move::Flags::EN_PASSANT_CAP)
        {
            const int cap_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
            update_nnue_halfka_both_perspectives<false>(white_king_sq, black_king_sq, Them, PAWN, cap_sq);
            touched_squares.push_back(cap_sq);
        }
        else if (to_piece != NO_PIECE)
        {
            update_nnue_halfka_both_perspectives<false>(white_king_sq, black_king_sq, Them, to_piece, to_sq);
        }

        // Zero-copy: collect the pre-move Full_Threats scoped feature set
        // directly from the live (still pre-move) board -- no full Board
        // snapshot needed, since touched_squares only depends on `move`
        // itself. See NnueEval::collect_threats_scoped.
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, old_white_threats, old_black_threats);

        eval_state.increment(move, Us);
        Board::play<Us>(move);

        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> new_white_threats, new_black_threats;
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, new_white_threats, new_black_threats);
        nnue_eval.template apply_threats_diff<WHITE>(old_white_threats, new_white_threats);
        nnue_eval.template apply_threats_diff<BLACK>(old_black_threats, new_black_threats);
#else
        eval_state.increment(move, Us);
        Board::play<Us>(move);
#endif
    }
    template <Color Us>
    inline void unplay(const Move move)
    {
#ifdef NNUE_EVAL
        const Piece from_piece = move.get_from_piece();
        if (from_piece == KING)
        {
            unplay_king_move<Us>(move);
            return;
        }

        constexpr Color Them = static_cast<Color>(!Us);
        nnue::threats::FixedIntList<nnue::threats::MAX_TOUCHED_SQUARES> touched_squares;
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> post_white_threats, post_black_threats;

        const int white_king_sq = king_sq[WHITE];
        const int black_king_sq = king_sq[BLACK];

        const int from_sq = move.get_from_sq();
        const int to_sq = move.get_to_sq();
        const uint32_t flags = move.get_flags();
        const Piece to_piece = move.get_to_piece();
        const Piece moved_piece = (flags == Move::Flags::PROMOTION_MASK) ? move.get_promo_piece() : from_piece;

        update_nnue_halfka_both_perspectives<false>(white_king_sq, black_king_sq, Us, moved_piece, to_sq);
        update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Us, from_piece, from_sq);

        touched_squares.push_back(from_sq);
        touched_squares.push_back(to_sq);

        if (flags == Move::Flags::EN_PASSANT_CAP)
        {
            const int cap_sq = (Us == WHITE) ? to_sq - 8 : to_sq + 8;
            update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Them, PAWN, cap_sq);
            touched_squares.push_back(cap_sq);
        }
        else if (to_piece != NO_PIECE)
        {
            update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Them, to_piece, to_sq);
        }

        // Zero-copy: collect the Full_Threats scoped feature set directly
        // from the live board while it still holds the post-move
        // (pre-unplay) state -- no full Board snapshot needed.
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, post_white_threats, post_black_threats);

        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);

        // apply_threats_diff removes features unique to its first
        // argument and adds features unique to its second, so passing
        // (post-move, pre-move) here correctly reverses the play()-time
        // update.
        nnue::threats::FixedIntList<nnue::threats::MAX_THREAT_FEATURES> pre_white_threats, pre_black_threats;
        nnue_eval.collect_threats_scoped_both(*this, touched_squares, pre_white_threats, pre_black_threats);
        nnue_eval.template apply_threats_diff<WHITE>(post_white_threats, pre_white_threats);
        nnue_eval.template apply_threats_diff<BLACK>(post_black_threats, pre_black_threats);
#else
        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);
#endif
    }

    inline EvalState &get_eval_state()
    {
        return eval_state;
    }

    inline const EvalState &get_eval_state() const
    {
        return eval_state;
    }
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
