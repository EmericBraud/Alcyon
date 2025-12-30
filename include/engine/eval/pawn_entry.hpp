#pragma once

#include "utils.hpp"

struct PawnEntry
{
    uint64_t key; // La pawn_key de EvalState
    int16_t mg;   // Score global des pions (mg_white - mg_black)
    int16_t eg;   // Score global des pions (eg_white - eg_black)
};

class PawnTable
{
    std::vector<PawnEntry> table;
    size_t index_mask;

public:
    uint64_t hits = 0;
    uint64_t misses = 0;

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

    bool probe(uint64_t key, int &mg, int &eg)
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

    void store(uint64_t key, int mg, int eg)
    {
        table[key & index_mask] = {key, (int16_t)mg, (int16_t)eg};
    }
    void reset_stats()
    {
        hits = 0;
        misses = 0;
    }

    double get_hit_rate() const
    {
        uint64_t total = hits + misses;
        if (total == 0)
            return 0.0;
        return (double)hits / total * 100.0;
    }
    PawnTable(size_t mb = 8)
    {
        resize(mb);
    }
};
