#pragma once
#include "core/move/move_list.hpp"

enum TTFlag : uint8_t
{
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

struct alignas(16) TTEntry
{
    std::atomic<uint64_t> key;
    // [Move: 32 bits | Score: 16 bits | Depth: 8 bits | Flag/Age: 8 bits]
    std::atomic<uint64_t> data;

    void save(uint64_t k, Move m, int16_t s, uint8_t d, uint8_t f)
    {
        uint64_t d_pack = (uint64_t)m.get_value();
        d_pack |= (uint64_t)(uint16_t)s << 32;
        d_pack |= (uint64_t)d << 48;
        d_pack |= (uint64_t)f << 56;

        // On écrit d'abord la donnée, puis la clé pour éviter qu'un autre thread
        // ne valide une clé correcte avec des données en cours d'écriture.
        data.store(d_pack, std::memory_order_relaxed);
        key.store(k, std::memory_order_relaxed);
    }

    // k_target est la clé recherchée.
    bool load(uint64_t k_target, Move &m, int16_t &s, uint8_t &d, uint8_t &f) const
    {
        if (key.load(std::memory_order_relaxed) == k_target)
        {
            uint64_t d_pack = data.load(std::memory_order_relaxed);
            m = Move(static_cast<uint32_t>(d_pack & 0xFFFFFFFF));
            s = static_cast<int16_t>((d_pack >> 32) & 0xFFFF);
            d = static_cast<uint8_t>((d_pack >> 48) & 0xFF);
            f = static_cast<uint8_t>((d_pack >> 56) & 0xFF);

            // Double vérification de la clé après lecture pour s'assurer
            // qu'elle n'a pas été modifiée pendant l'extraction du pack.
            return key.load(std::memory_order_relaxed) == k_target;
        }
        return false;
    }

    TTEntry() : key(0), data(0) {}
};

struct TTBucket
{
    TTEntry entries[4];
};

class TranspositionTable
{
private:
    std::unique_ptr<TTBucket[]> table;
    size_t bucket_count = 0;
    size_t index_mask = 0;
    uint8_t current_age = 0;

    int score_to_tt(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score + ply;
        if (score < -MATE_SCORE + 1000)
            return score - ply;
        return score;
    }

    int score_from_tt(int score, int ply)
    {
        if (score > MATE_SCORE - 1000)
            return score - ply;
        if (score < -MATE_SCORE + 1000)
            return score + ply;
        return score;
    }

public:
    TranspositionTable() : table(nullptr) {}

    void resize(size_t mb_size)
    {
        size_t bytes = mb_size * 1024 * 1024;
        size_t desired_buckets = bytes / sizeof(TTBucket);

        size_t n = 1;
        while (n <= desired_buckets)
            n <<= 1;
        n >>= 1;

        // Allocation manuelle via unique_ptr
        // Cela appelle le constructeur par défaut de TTEntry (key=0, data=0)
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
                table[i].entries[j].key.store(0, std::memory_order_relaxed);
                table[i].entries[j].data.store(0, std::memory_order_relaxed);
            }
        }
    }

    void next_generation() { current_age = (current_age + 4) & 0xFC; }

    void store(uint64_t key, int depth, int ply, int score, uint8_t flag, Move move)
    {
        TTBucket &bucket = table[key & index_mask];
        int replace_idx = 0;
        int min_priority = 1000000;

        for (int i = 0; i < 4; ++i)
        {
            Move m_old;
            int16_t s_old;
            uint8_t d_old;
            uint8_t f_old;

            // Tentative de chargement pour vérifier si la clé existe déjà
            if (bucket.entries[i].load(key, m_old, s_old, d_old, f_old))
            {
                bool old_generation = ((f_old ^ current_age) & 0xFC) != 0;

                // On met à jour si profondeur supérieure ou ancienne recherche
                if (depth >= d_old || old_generation)
                {
                    if (move == 0)
                        move = m_old;
                    bucket.entries[i].save(key, move, (int16_t)score_to_tt(score, ply), (uint8_t)depth, flag | current_age);
                }
                return;
            }

            // 1. On charge la clé une seule fois
            uint64_t k_tmp = bucket.entries[i].key.load(std::memory_order_relaxed);

            // OPTIMISATION : Si le slot est vide, on le prend immédiatement !
            if (k_tmp == 0)
            {
                replace_idx = i;
                break; // On arrête la recherche, on a trouvé le slot parfait
            }

            // 2. Si ce n'est pas le bon slot et qu'il n'est pas vide, on calcule sa priorité
            uint64_t d_tmp = bucket.entries[i].data.load(std::memory_order_relaxed);
            uint8_t entry_depth = (uint8_t)((d_tmp >> 48) & 0xFF);
            uint8_t entry_flag = (uint8_t)((d_tmp >> 56) & 0xFF);

            bool old = ((entry_flag ^ current_age) & 0xFC) != 0;
            int priority = entry_depth;

            if (!old)
                priority += 100; // Les entrées de la génération actuelle sont protégées

            if (priority < min_priority)
            {
                min_priority = priority;
                replace_idx = i;
            }
        }

        bucket.entries[replace_idx].save(key, move, (int16_t)score_to_tt(score, ply), (uint8_t)depth, flag | current_age);
    }

    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move)
    {
        TTBucket &bucket = table[key & index_mask];

        for (int i = 0; i < 4; ++i)
        {
            Move m;
            int16_t s;
            uint8_t d;
            uint8_t f;
            if (bucket.entries[i].load(key, m, s, d, f))
            {
                best_move = m;
                if (d >= depth)
                {
                    int score = score_from_tt(s, ply);
                    uint8_t flag = f & 0x03;

                    if (flag == TT_EXACT)
                    {
                        return_score = score;
                        return true;
                    }
                    if (flag == TT_ALPHA && score <= alpha)
                    {
                        return_score = alpha;
                        return true;
                    }
                    if (flag == TT_BETA && score >= beta)
                    {
                        return_score = beta;
                        return true;
                    }
                }
                return false;
            }
        }
        return false;
    }

    Move get_move(uint64_t key)
    {
        TTBucket &bucket = table[key & index_mask];
        for (int i = 0; i < 4; ++i)
        {
            Move m;
            int16_t s;
            uint8_t d;
            uint8_t f;
            if (bucket.entries[i].load(key, m, s, d, f))
                return m;
        }
        return 0;
    }

    int get_hashfull() const
    {
        size_t count = 0;
        size_t sample = std::min(bucket_count, (size_t)1000);
        for (size_t i = 0; i < sample; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                if (table[i].entries[j].key.load(std::memory_order_relaxed) != 0)
                    count++;
            }
        }
        return (int)(count * 1000 / (sample * 4));
    }

    inline void prefetch(uint64_t hash)
    {
        _mm_prefetch((const char *)&table[hash & index_mask], _MM_HINT_T0);
    }
};