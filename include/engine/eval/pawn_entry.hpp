#pragma once

#include <cstdint>
#include <vector>

#include "core/utils/mask.hpp"

struct PawnEntry
{
    U64 key;
    std::int16_t mg;
    std::int16_t eg;
};

class PawnTable
{
    std::vector<PawnEntry> table;
    size_t index_mask;

public:
    U64 hits = 0;
    U64 misses = 0;

    void resize(size_t mb)
    {
        size_t n = (mb * 1024 * 1024) / sizeof(PawnEntry);
        size_t size = 1;
        while (size <= n)
            size <<= 1;
        size >>= 1;
        table.assign(size, {0, 0, 0});
        index_mask = size - 1;
    }

    bool probe(U64 key, int &mg, int &eg)
    {
        PawnEntry &entry = table[key & index_mask];
        if (entry.key == key)
        {
            mg = entry.mg;
            eg = entry.eg;
            hits++;
            return true;
        }
        misses++;
        return false;
    }

    void store(U64 key, int mg, int eg)
    {
        table[key & index_mask] = {key, (std::int16_t)mg, (std::int16_t)eg};
    }
    void reset_stats()
    {
        hits = 0;
        misses = 0;
    }

    double get_hit_rate() const
    {
        U64 total = hits + misses;
        if (total == 0)
            return 0.0;
        return (double)hits / total * 100.0;
    }
    PawnTable(size_t mb = 8)
    {
        resize(mb);
    }
};
