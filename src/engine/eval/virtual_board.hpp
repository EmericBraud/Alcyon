#pragma once

// This file aims at creating a board that also implements automatic increment / decrement
// of an Eval State to make the lazy evaluation almost free

#include "core/piece/color.hpp"
#include "core/board/board.hpp"
#include "engine/eval/move_eval_increment.hpp"

class VBoard : public Board
{
    EvalState eval_state;

public:
    VBoard &operator=(const VBoard &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

            eval_state = other.eval_state;
        };

        return *this;
    }
    VBoard &operator=(const Board &other)
    {
        if (this != &other)
        {
            Board::operator=(other);

            eval_state = EvalState(pieces_occ);
        };

        return *this;
    }
    VBoard(const VBoard &other)
        : Board(other),
          eval_state(other.eval_state)
    {
    }
    VBoard(const Board &other)
        : Board(other),
          eval_state(other.get_all_bitboards())
    {
    }

    VBoard(VBoard &&other) noexcept
        : Board(std::move(other)),
          eval_state(std::move(other.eval_state))
    {
    }

    VBoard &operator=(VBoard &&other) noexcept
    {
        if (this != &other)
        {
            Board::operator=(std::move(other));
            eval_state = std::move(other.eval_state);
        }
        return *this;
    }

    VBoard() : Board(), eval_state(Board::get_all_bitboards())
    {
    }

    bool load_fen(const std::string_view fen_string)
    {
        bool r = Board::load_fen(fen_string);
        eval_state = EvalState(get_all_bitboards());
        return r;
    }

    inline void play(const Move move)
    {
        Color Us = get_side_to_move();
        eval_state.increment(move, Us);
        Board::play(move);
    }
    inline void unplay(const Move move)
    {
        Color Us = !get_side_to_move();
        Board::unplay(move);
        eval_state.decrement(move, Us);
    }

    template <Color Us>
    inline void play(const Move move)
    {
        eval_state.increment(move, Us);
        Board::play<Us>(move);
    }
    template <Color Us>
    inline void unplay(const Move move)
    {
        Board::unplay<Us>(move);
        eval_state.decrement(move, Us);
    }

    inline EvalState &get_eval_state()
    {
        return eval_state;
    }

    inline const EvalState &get_eval_state() const
    {
        return eval_state;
    }
};