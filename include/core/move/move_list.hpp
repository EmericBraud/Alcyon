#pragma once

#include "move.hpp"

#define MAX_MOVES 256

struct MoveList
{
    Move moves[MAX_MOVES];
    int scores[MAX_MOVES];

    int count = 0;

    Move *begin() { return &moves[0]; }
    Move *end() { return &moves[count]; }

    MoveList() {}

    inline void push(Move m)
    {
        assert(count < 256);
        moves[count++] = m;
    }

    Move &operator[](int index) { return moves[index]; }

    inline Move &pick_best_move(int from_i)
    {
        int best_index = from_i;
        for (int j = from_i; j < count; ++j)
        {
            if (scores[j] > scores[best_index])
            {
                best_index = j;
            }
        }
        std::swap(moves[from_i], moves[best_index]);
        std::swap(scores[from_i], scores[best_index]);
        return moves[from_i];
    }
    inline int size() const
    {
        return count;
    }

    inline bool empty() const
    {
        return count == 0;
    }
};