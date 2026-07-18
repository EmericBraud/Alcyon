#pragma once

// This file aims at creating a board that also implements automatic increment / decrement
// of an Eval State to make the lazy evaluation almost free

#include "core/piece/color.hpp"
#include "core/board/board.hpp"
#include "engine/eval/hce/move_eval_increment.hpp"
#include "common/file.hpp"
#ifdef NNUE_EVAL
#include <vector>
#include "engine/eval/nnue/nnue_eval.hpp"
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

    template <Color Us>
    inline void play(const Move move)
    {
#ifdef NNUE_EVAL
        constexpr Color Them = static_cast<Color>(!Us);
        const Piece from_piece = move.get_from_piece();
        Board board_before;
        std::vector<int> touched_squares;
        bool do_threats = false;

        if (from_piece != KING)
        {
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

            touched_squares = {from_sq, to_sq};

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

            board_before = *this;
            do_threats = true;
        }
#endif

        eval_state.increment(move, Us);
        Board::play<Us>(move);

#ifdef NNUE_EVAL
        if (from_piece == KING)
            nnue_eval.initialize(*this);
        else if (do_threats)
        {
            nnue_eval.template update_threats_scoped<WHITE>(board_before, *this, touched_squares);
            nnue_eval.template update_threats_scoped<BLACK>(board_before, *this, touched_squares);
        }
#endif
    }
    template <Color Us>
    inline void unplay(const Move move)
    {
#ifdef NNUE_EVAL
        constexpr Color Them = static_cast<Color>(!Us);
        const Piece from_piece = move.get_from_piece();
        Board board_before;
        std::vector<int> touched_squares;
        bool do_threats = false;

        if (from_piece != KING)
        {
            const int white_king_sq = king_sq[WHITE];
            const int black_king_sq = king_sq[BLACK];

            const int from_sq = move.get_from_sq();
            const int to_sq = move.get_to_sq();
            const uint32_t flags = move.get_flags();
            const Piece to_piece = move.get_to_piece();
            const Piece moved_piece = (flags == Move::Flags::PROMOTION_MASK) ? move.get_promo_piece() : from_piece;

            update_nnue_halfka_both_perspectives<false>(white_king_sq, black_king_sq, Us, moved_piece, to_sq);
            update_nnue_halfka_both_perspectives<true>(white_king_sq, black_king_sq, Us, from_piece, from_sq);

            touched_squares = {from_sq, to_sq};

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

            board_before = *this; // snapshot of the post-move (pre-unplay) board
            do_threats = true;
        }
#endif

        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);

#ifdef NNUE_EVAL
        if (from_piece == KING)
            nnue_eval.initialize(*this);
        else if (do_threats)
        {
            // update_threats_scoped diffs feature sets between two board snapshots
            // regardless of direction, so passing (post-move, pre-move) here
            // correctly reverses the play()-time update.
            nnue_eval.template update_threats_scoped<WHITE>(board_before, *this, touched_squares);
            nnue_eval.template update_threats_scoped<BLACK>(board_before, *this, touched_squares);
        }
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
