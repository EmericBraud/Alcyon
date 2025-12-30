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
    uint64_t key;
    Move move;
    int16_t score;
    uint8_t depth;
    uint8_t flag; // [7:2] Age (6 bits), [1:0] Type (2 bits)

    TTEntry() : key(0), move(0), score(0), depth(0), flag(0) {}
};

struct TTBucket
{
    TTEntry entries[4];
};

class TranspositionTable
{
private:
    std::vector<TTBucket> table;
    size_t bucket_count;
    size_t index_mask;
    uint8_t current_age = 0; // Utilise les 6 bits de poids fort

    // Normalisation des scores de mat pour la TT
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
    void resize(size_t mb_size)
    {
        size_t bytes = mb_size * 1024 * 1024;
        size_t desired_buckets = bytes / sizeof(TTBucket);

        // Puissance de 2 pour un masquage rapide
        size_t n = 1;
        while (n <= desired_buckets)
            n <<= 1;
        n >>= 1;

        table.assign(n, TTBucket{});
        index_mask = n - 1;
        bucket_count = n;
    }

    void clear()
    {
        std::fill(table.begin(), table.end(), TTBucket{});
    }

    // Appelé à chaque début de nouvelle recherche (itération de l'ID)
    // On incrémente de 4 car on stocke l'âge dans les bits [7:2]
    void next_generation() { current_age = (current_age + 4) & 0xFC; }

    void store(uint64_t key, int depth, int ply, int score, uint8_t flag, Move move)
    {
        TTBucket &bucket = table[key & index_mask];

        int best_idx = 0;
        int worst_priority = 1000000;

        for (int i = 0; i < 4; ++i)
        {
            // 1. Hit : On a déjà la clé
            if (bucket.entries[i].key == key)
            {
                // On n'écrase que si la nouvelle profondeur est meilleure
                // ou si l'entrée existante appartient à une ancienne recherche
                bool old = ((bucket.entries[i].flag ^ current_age) & 0xFC) != 0;
                if (depth >= bucket.entries[i].depth || old)
                {
                    if (move != 0)
                        bucket.entries[i].move = move;
                    bucket.entries[i].score = (int16_t)score_to_tt(score, ply);
                    bucket.entries[i].depth = (uint8_t)depth;
                    bucket.entries[i].flag = flag | current_age;
                }
                return;
            }

            // 2. Stratégie de remplacement (Priority = Depth + Age_Bonus)
            // Une entrée d'une génération précédente est très facile à remplacer
            bool old = ((bucket.entries[i].flag ^ current_age) & 0xFC) != 0;
            int priority = bucket.entries[i].depth;
            if (!old)
                priority += 100; // Bonus de survie pour la génération actuelle

            if (priority < worst_priority)
            {
                worst_priority = priority;
                best_idx = i;
            }
        }

        // 3. Remplacement
        bucket.entries[best_idx].key = key;
        bucket.entries[best_idx].move = move;
        bucket.entries[best_idx].score = (int16_t)score_to_tt(score, ply);
        bucket.entries[best_idx].depth = (uint8_t)depth;
        bucket.entries[best_idx].flag = flag | current_age;
    }

    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move)
    {
        TTBucket &bucket = table[key & index_mask];

        for (int i = 0; i < 4; ++i)
        {
            if (bucket.entries[i].key == key)
            {
                TTEntry &entry = bucket.entries[i];
                best_move = entry.move;

                if (entry.depth >= depth)
                {
                    int score = score_from_tt(entry.score, ply);
                    uint8_t flag = entry.flag & 0x03;

                    if (flag == TT_EXACT)
                    {
                        return_score = score;
                        return true;
                    }
                    if (flag == TT_ALPHA && score <= alpha)
                    {
                        return_score = alpha;
                        return true; // On retourne alpha (borne sup)
                    }
                    if (flag == TT_BETA && score >= beta)
                    {
                        return_score = beta;
                        return true; // On retourne beta (borne inf)
                    }
                }
                return false; // Clé trouvée mais profondeur insuffisante
            }
        }
        return false;
    }

    // Utilisé par le moteur pour le tri des coups (Ordering)
    Move get_move(uint64_t key)
    {
        TTBucket &bucket = table[key & index_mask];
        for (int i = 0; i < 4; ++i)
        {
            if (bucket.entries[i].key == key)
                return bucket.entries[i].move;
        }
        return 0;
    }

    // Retourne le taux de remplissage en permillage (0-1000)
    // On vérifie les 1000 premiers buckets pour une estimation rapide
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

    inline void prefetch(uint64_t hash)
    {
        _mm_prefetch((const char *)&table[hash & index_mask], _MM_HINT_T0);
    }
};