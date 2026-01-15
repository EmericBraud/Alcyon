#pragma once
#include <cstdint>
#include <memory>

#include "common/cpu.hpp"
#include "core/move/move.hpp"
#include "engine/config/config.hpp"

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
        if (score > engine::config::eval::MateScore - 256)
            return score + ply;
        if (score < -engine::config::eval::MateScore + 256)
            return score - ply;
        return score;
    }

    int score_from_tt(int score, int ply)
    {
        if (score > engine::config::eval::MateScore - 256)
            return score - ply;
        if (score < -engine::config::eval::MateScore + 256)
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
                    bool new_is_mate = abs(score) > engine::config::eval::MateScore - 256;
                    bool old_is_mate = abs(s_prev) > engine::config::eval::MateScore - 256;

                    bool replace;

                    if (new_is_mate && old_is_mate)
                        replace = abs(score) < abs(s_prev); // PLUS COURT = MEILLEUR
                    else
                        replace = depth >= d_prev || old_gen;
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

            int priority = d_old + (((f_old ^ current_age) & 0xFC) ? 0 : 100);
            if (priority < min_priority)
            {
                min_priority = priority;
                replace_idx = i;
            }
        }

        if (replace_idx != -1)
            bucket.entries[replace_idx].save(key, move, (int16_t)score_to_tt(score, ply), (std::uint8_t)depth, flag | current_age);
    }

    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move, TTFlag &flag)
    {
        TTBucket &bucket = table[key & index_mask];
        bool found_move = false;

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
            }

            if (d < depth)
                continue;

            int score = score_from_tt(s, ply);
            flag = static_cast<TTFlag>(f & 0x03);

            if (flag == TT_EXACT)
            {
                return_score = score;
                return true;
            }
            if (flag == TT_ALPHA && score <= alpha)
            {
                return_score = score;
                return true;
            }
            if (flag == TT_BETA && score >= beta)
            {
                return_score = score;
                return true;
            }
        }

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