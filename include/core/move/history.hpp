#pragma once

#include <cassert>

#include "core/utils/constants.hpp"
#include "core/utils/mask.hpp"
#include "core/move/move.hpp"

struct alignas(16) UndoInfo
{
    U64 zobrist_key;
    uint16_t halfmove_clock;
    int last_irreversible_index;
    Move move;
    int en_passant_sq;
    uint8_t castling_rights;
};

struct History
{
    UndoInfo list[core::constants::MaxHistorySize];
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