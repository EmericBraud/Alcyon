#pragma once
#include "engine/eval/move_eval_increment.hpp"
#define MAX_ELE 256

struct alignas(16) UndoInfo
{
    U64 zobrist_key;
    uint16_t halfmove_clock;
    int last_irreversible_index;
    Move move;
};

struct History
{
    UndoInfo list[MAX_ELE];
    size_t count = 0;

    inline void push_back(const UndoInfo &u)
    {
        list[count++] = u;
    }

    inline void pop_back()
    {
        --count;
    }

    inline const UndoInfo &back() const
    {
        assert(count > 0);
        return list[count - 1];
    }

    inline UndoInfo &operator[](int index)
    {
        return list[index];
    }

    inline const UndoInfo &operator[](int index) const
    {
        return list[index];
    }

    inline UndoInfo *begin()
    {
        return &list[0];
    }

    inline UndoInfo *end()
    {
        return &list[count];
    }

    inline size_t size() const
    {
        return count;
    }

    inline void clear()
    {
        count = 0;
    }

    inline bool empty() const
    {
        return count == 0;
    }
};