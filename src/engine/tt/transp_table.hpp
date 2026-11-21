#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "common/cpu.hpp"
#include "core/move/move.hpp"
#include "engine/config/config.hpp"

#ifdef ALCYON_TT_STATS
#include <cstdio>
#include <vector>
// Instrumentation de la TT (build separe, -DALCYON_TT_STATS) : utilisation
// reelle des entrees. Mono-thread (Threads=1 pour les mesures).
struct TTStats
{
    unsigned long long probes = 0, probe_hits = 0, probe_cuts = 0;
    unsigned long long stores = 0, store_empty = 0, store_same = 0, store_evict = 0, store_skip = 0;
    // Entrees evincees : [classe de profondeur][classe de lectures]
    unsigned long long evicted[5][6] = {};
    unsigned long long evicted_old_gen = 0;
    // Lectures par emplacement du paquet (0..3) : les slots sont-ils egaux ?
    unsigned long long slot_hits[4] = {};
    std::vector<std::uint16_t> reads;
    static int dclass(int d) { return d <= 0 ? 0 : d <= 2 ? 1 : d <= 5 ? 2 : d <= 9 ? 3 : 4; }
    static int rclass(int r) { return r == 0 ? 0 : r == 1 ? 1 : r <= 3 ? 2 : r <= 7 ? 3 : r <= 31 ? 4 : 5; }
    ~TTStats()
    {
        std::fprintf(stderr, "TTSTATS probes %llu hits %llu cuts %llu\n", probes, probe_hits, probe_cuts);
        std::fprintf(stderr, "TTSTATS stores %llu empty %llu same %llu evict %llu skip %llu evict_old_gen %llu\n",
                     stores, store_empty, store_same, store_evict, store_skip, evicted_old_gen);
        std::fprintf(stderr, "TTSTATS slot_hits %llu %llu %llu %llu\n", slot_hits[0], slot_hits[1], slot_hits[2], slot_hits[3]);
        for (int d = 0; d < 5; ++d)
            std::fprintf(stderr, "TTSTATS evicted_d%d %llu %llu %llu %llu %llu %llu\n", d, evicted[d][0], evicted[d][1],
                         evicted[d][2], evicted[d][3], evicted[d][4], evicted[d][5]);
    }
};
inline TTStats tt_stats;
#define TT_STAT(x) x
#else
#define TT_STAT(x)
#endif

// NOTE: probe() used to consult an asynchronous GPU-eval score cache and
// override its own score with it. That whole subsystem was removed after
// measurement (no positive Elo in any of its three integrations) -- its
// results live on in docs/gpu-async-eval/, and the code at the
// gpu-eval-experiment tag.

enum TTFlag : std::uint8_t
{
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

struct TTEntry
{
    uint64_t key;
    uint64_t data;
    void save(uint64_t k, Move m, int16_t s, std::uint8_t d, std::uint8_t f)
    {
        uint64_t d_pack = (uint64_t)m.get_value() |
                          ((uint64_t)(uint16_t)s << 32) |
                          ((uint64_t)d << 48) |
                          ((uint64_t)f << 56);

        data = d_pack ^ k; // XOR Trick
        key = k;
    }

    bool load(uint64_t k_target, Move &m, int16_t &s, std::uint8_t &d, std::uint8_t &f) const
    {
        const uint64_t k = key;

        if (k == k_target)
        {
            const uint64_t d_xor = data;
            const uint64_t d_pack = d_xor ^ k;
            m = Move(static_cast<uint32_t>(d_pack & 0xFFFFFFFF));
            s = static_cast<int16_t>((d_pack >> 32) & 0xFFFF);
            d = static_cast<std::uint8_t>((d_pack >> 48) & 0xFF);
            f = static_cast<std::uint8_t>((d_pack >> 56) & 0xFF);
            return true;
        }
        return false;
    }

    TTEntry() : key(0), data(0) {}
};

struct alignas(64) TTBucket
{
    TTEntry entries[4];
};

class TranspositionTable
{
private:
    std::unique_ptr<TTBucket[]> table;
    size_t bucket_count = 0;
    size_t index_mask = 0;
    std::uint8_t current_age = 0;

    int score_to_tt(int score, int ply)
    {
        if (score > engine_constants::eval::MateScore - 256)
            return score + ply;
        if (score < -engine_constants::eval::MateScore + 256)
            return score - ply;
        return score;
    }

    int score_from_tt(int score, int ply)
    {
        if (score > engine_constants::eval::MateScore - 256)
            return score - ply;
        if (score < -engine_constants::eval::MateScore + 256)
            return score + ply;
        return score;
    }

public:
    TranspositionTable() : table(nullptr) {}

    void resize(size_t mb_size)
    {
        size_t bytes = mb_size * 1024 * 1024;
        size_t n = 1;
        while (n * sizeof(TTBucket) <= bytes)
            n <<= 1;
        n >>= 1;

        table = std::make_unique<TTBucket[]>(n);
        index_mask = n - 1;
        bucket_count = n;
        TT_STAT(tt_stats.reads.assign(n * 4, 0));
    }

    void clear()
    {
        if (!table)
            return;
        for (size_t i = 0; i < bucket_count; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                table[i].entries[j].key = 0;
                table[i].entries[j].data = 0;
            }
        }
    }

    void next_generation() { current_age = (current_age + 4) & 0xFC; }

    void store(uint64_t key, int depth, int ply, int score, std::uint8_t flag, Move move)
    {
        TTBucket &bucket = table[key & index_mask];
        TT_STAT(++tt_stats.stores);
        TT_STAT(const size_t base_idx = (key & index_mask) * 4);
        int replace_idx = -1;
        int min_priority = 1000000;

        for (int i = 0; i < 4; ++i)
        {
            // Lecture directe non-atomique
            uint64_t k_old = bucket.entries[i].key;

            if (k_old == key)
            {
                Move m_prev;
                int16_t s_prev;
                std::uint8_t d_prev;
                std::uint8_t f_prev;
                if (bucket.entries[i].load(key, m_prev, s_prev, d_prev, f_prev))
                {
                    bool old_gen = ((f_prev ^ current_age) & 0xFC) != 0;
                    bool new_is_mate = abs(score) > engine_constants::eval::MateScore - 256;
                    bool old_is_mate = abs(s_prev) > engine_constants::eval::MateScore - 256;

                    bool replace;

                    if (new_is_mate && old_is_mate)
                        replace = abs(score) < abs(s_prev); // PLUS COURT = MEILLEUR
                    else
                        replace = depth >= d_prev || old_gen;
                    TT_STAT(if (replace) ++tt_stats.store_same; else ++tt_stats.store_skip);
                    if (replace)
                    {
                        bucket.entries[i].save(key, (move != 0) ? move : m_prev,
                                               (int16_t)score_to_tt(score, ply), (std::uint8_t)depth, flag | current_age);
                    }
                }
                return;
            }

            if (k_old == 0)
            {
                replace_idx = i;
                break;
            }

            uint64_t d_pack = bucket.entries[i].data ^ k_old;
            std::uint8_t d_old = (std::uint8_t)((d_pack >> 48) & 0xFF);
            std::uint8_t f_old = (std::uint8_t)((d_pack >> 56) & 0xFF);

            int priority;
            if (engine_constants::search::TtReplace == 0)
                priority = d_old + (((f_old ^ current_age) & 0xFC) ? 0 : 100);
            else
            {
                // Age relatif (en coups) : une entree profonde d'un coup recent
                // survit a une entree peu profonde du coup en cours.
                const int age_diff = ((current_age - (f_old & 0xFC)) & 0xFC) >> 2;
                priority = d_old - engine_constants::search::TtAgeWeight * age_diff;
            }
            if (priority < min_priority)
            {
                min_priority = priority;
                replace_idx = i;
            }
        }

        // Les entrees de profondeur <= 0 (80 % jamais relues) n'evincent qu'une
        // place vide ou une entree d'un coup precedent.
        if (replace_idx != -1 && engine_constants::search::TtShallowGuard && depth <= 0)
        {
            const TTEntry &v = bucket.entries[replace_idx];
            if (v.key != 0 && !((((v.data ^ v.key) >> 56) ^ current_age) & 0xFC))
                replace_idx = -1;
        }
#ifdef ALCYON_TT_STATS
        if (replace_idx != -1)
        {
            TTEntry &e = bucket.entries[replace_idx];
            std::uint16_t &r = tt_stats.reads[base_idx + replace_idx];
            if (e.key == 0)
                ++tt_stats.store_empty;
            else
            {
                const uint64_t dp = e.data ^ e.key;
                const int d_old = int((dp >> 48) & 0xFF);
                const std::uint8_t f_old = std::uint8_t((dp >> 56) & 0xFF);
                ++tt_stats.store_evict;
                ++tt_stats.evicted[TTStats::dclass(d_old)][TTStats::rclass(r)];
                if ((f_old ^ current_age) & 0xFC)
                    ++tt_stats.evicted_old_gen;
            }
            r = 0;
        }
#endif
        if (replace_idx != -1)
            bucket.entries[replace_idx].save(key, move, (int16_t)score_to_tt(score, ply), (std::uint8_t)depth, flag | current_age);
    }

    // Lecture seule de l'entree la plus profonde pour cette cle, sans les
    // regles de coupure de probe(). Sert au dump de docs/learned-pruning.md :
    // au point de decision probe() a deja echoue, mais une entree trop
    // courte ou dont la borne ne coupait pas reste une information.
    bool peek(uint64_t key, int ply, int &score, int &depth, TTFlag &flag)
    {
        TTBucket &bucket = table[key & index_mask];
        bool found = false;
        for (int i = 0; i < 4; ++i)
        {
            Move m;
            int16_t s;
            std::uint8_t d;
            std::uint8_t f;
            if (!bucket.entries[i].load(key, m, s, d, f) || (found && d <= depth))
                continue;
            score = score_from_tt(s, ply);
            depth = d;
            flag = static_cast<TTFlag>(f & 0x03);
            found = true;
        }
        return found;
    }

    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move, TTFlag &flag)
    {
        TTBucket &bucket = table[key & index_mask];
        bool found_move = false;
        TT_STAT(++tt_stats.probes);

        for (int i = 0; i < 4; ++i)
        {
            Move m;
            int16_t s;
            std::uint8_t d;
            std::uint8_t f;
            if (!bucket.entries[i].load(key, m, s, d, f))
                continue;

            if (!found_move)
            {
                best_move = m;
                found_move = true;
                TT_STAT(++tt_stats.probe_hits);
                TT_STAT(++tt_stats.slot_hits[i]);
                TT_STAT({ auto &r = tt_stats.reads[(key & index_mask) * 4 + i]; if (r < 65535) ++r; });
            }

            if (d < depth)
                continue;

            int score = score_from_tt(s, ply);
            flag = static_cast<TTFlag>(f & 0x03);

            // NOTE: a GPU-eval score used to override `score` here, for a
            // position the main TT ALSO had an entry for. Removed, for two
            // independent reasons:
            //  - Unsound: the bound checks below (TT_ALPHA/TT_BETA) apply
            //    THIS entry's bound flag, which only ever certified the
            //    entry's OWN search score ("true value <= s"). That
            //    guarantee does not transfer to a substituted static CNN
            //    score, so the cutoffs it produced were unjustified.
            //  - Pointless: to even reach here the entry must have passed
            //    `d < depth`, i.e. it holds a real search result at least
            //    as deep as what this node is being asked for -- strictly
            //    better information than a static eval.
            // The GPU score is consulted where it actually helps instead:
            // the frontier-node case in negamax.cpp's should_qsearch
            // branch, where there is no search result at all yet.

            if (flag == TT_EXACT || (flag == TT_ALPHA && score <= alpha) || (flag == TT_BETA && score >= beta))
            {
                TT_STAT(++tt_stats.probe_cuts);
                return_score = score;
                return true;
            }
        }

        // NOTE: the brand-new-node case (no main-TT entry at all, e.g. a
        // PV-leaf child freshly precomputed by the GPU queue) is handled
        // in negamax.cpp's should_qsearch branch, not here -- that's the
        // only place where the absence of any search result makes a
        // static GPU score the best available answer. See
        // docs/gpu-async-eval/consultative-eval-measurements.md.
        return false;
    }

    Move get_move(uint64_t key) const
    {
        TTBucket &bucket = table[key & index_mask];
        for (int i = 0; i < 4; ++i)
        {
            Move m;
            int16_t s;
            std::uint8_t d;
            std::uint8_t f;
            if (bucket.entries[i].load(key, m, s, d, f))
                return m;
        }
        return Move(0);
    }

    int get_hashfull() const
    {
        size_t count = 0;
        size_t sample = std::min(bucket_count, (size_t)1000);
        for (size_t i = 0; i < sample; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (table[i].entries[j].key != 0)
                    count++;
            }
        }
        return (int)(count * 1000 / (sample * 4));
    }

    inline void prefetch(uint64_t hash) const
    {
        cpu::prefetch<TTBucket, true, 1>(&table[hash & index_mask]);
    }
};