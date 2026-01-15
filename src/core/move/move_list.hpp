#pragma once

#include "common/constants.hpp"
#include "move.hpp"

struct MoveList
{
    Move moves[constants::MaxMoves];
    int scores[constants::MaxMoves];
    bool is_tactical[constants::MaxMoves];

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
        std::swap(is_tactical[from_i], is_tactical[best_index]);
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