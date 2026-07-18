#pragma once

// Fixed-capacity, stack-allocated replacement for std::vector<int> in the
// Full_Threats scoped-recompute hot path (see full_threats_incremental.hpp
// and NnueEval::collect_threats_scoped/apply_threats_diff). Every search node
// that plays/unplays a non-king move drives several of these; std::vector's
// heap allocation/reallocation (and the memmove that comes with it) showed up
// heavily in profiling. All calls in that path bound the number of entries
// they can produce (see MAX_THREAT_FEATURES / MAX_TOUCHED_SQUARES in
// full_threats_encoder.hpp / full_threats_incremental.hpp), so a fixed-size
// std::array plus a running count is sufficient -- no heap ever touched.
//
// push_back() intentionally FATALs (hard abort) rather than silently
// truncating on overflow: silently dropping a feature index here would
// desync the accumulator from the true board state, which is a correctness
// bug and strictly worse than the performance problem this class fixes.

#include <array>
#include <cstddef>

#include "common/fatal.hpp"

namespace nnue::threats
{
    template <int Capacity>
    struct FixedIntList
    {
        // Deliberately NOT value-initialized ({}): zero-filling the whole
        // array on every construction (memset over Capacity ints, up to 8KB
        // for MAX_THREAT_FEATURES=2048) showed up as the top profiling hotspot
        // once heap allocation was removed from this hot path. Safe because
        // every accessor (operator[], begin/end, push_back) is bounded by
        // `count`, which only ever exposes slots that were actually written.
        std::array<int, Capacity> data;
        int count = 0;

        inline void push_back(int value)
        {
            if (count >= Capacity)
                FATAL("FixedIntList<" << Capacity << "> overflow: attempted to push a "
                                       << (count + 1) << "th element");
            data[count++] = value;
        }

        inline void clear() { count = 0; }
        inline int size() const { return count; }
        inline bool empty() const { return count == 0; }

        inline int operator[](std::size_t i) const { return data[i]; }
        inline int &operator[](std::size_t i) { return data[i]; }

        inline int *begin() { return data.data(); }
        inline int *end() { return data.data() + count; }
        inline const int *begin() const { return data.data(); }
        inline const int *end() const { return data.data() + count; }
    };
}
