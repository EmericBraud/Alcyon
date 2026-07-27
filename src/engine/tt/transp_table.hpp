#pragma once
#include <cstdint>
#include <memory>

#include "common/cpu.hpp"
#include "common/huge_pages.hpp"
#include "core/move/move.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/config/config.hpp"

enum TTFlag : std::uint8_t
{
    TT_EXACT = 0,
    TT_ALPHA = 1,
    TT_BETA = 2
};

// Sentinelle "pas d'éval statique stockée" : les stores issus de negamax
// n'ont pas d'éval complète sous la main (ils n'utilisent que l'estimation
// PSQT), et une position en échec n'a pas d'éval statique exploitable.
constexpr int16_t TT_NO_EVAL = INT16_MIN;

// Le Move complet (32 bits : flags, pièce qui bouge, pièce capturée...) ne
// tiendrait pas dans `data` à côté de l'éval statique. On ne stocke que
// l'information non reconstructible depuis la position : from + to +
// pièce de promotion, 14 bits. Tout le reste est redéduit au probe par
// decompress_tt_move() ci-dessous, via exactement le même chemin que le
// générateur de coups (init_move_flags + règle du roque), de sorte que le
// move reconstruit est BIT-IDENTIQUE à celui que la génération produira --
// la déduplication du MovePicker (comparaison de valeur 32 bits) continue
// de fonctionner.
inline uint16_t compress_tt_move(Move m)
{
    if (m == 0)
        return 0;
    return static_cast<uint16_t>(m.get_from_sq() |
                                 (m.get_to_sq() << 6) |
                                 (((m.get_promo_piece() - KNIGHT) & 0x3) << 12));
}

inline Move decompress_tt_move(const Board &board, uint16_t c)
{
    if (c == 0)
        return Move(0);
    const int from = c & 0x3F;
    const int to = (c >> 6) & 0x3F;
    const Piece from_piece = board.get_p(from);
    // Un pion qui arrive sur la dernière rangée promeut forcément -- même
    // critère que init_move_flags. Hors promotion, le champ prom du Move
    // vaut QUEEN par défaut (convention du constructeur, voir move.hpp) :
    // on le reproduit pour rester bit-identique aux coups générés.
    const bool is_promo = (from_piece == PAWN) && (to / 8 % 7 == 0);
    const Piece prom = is_promo ? static_cast<Piece>(KNIGHT + ((c >> 12) & 0x3)) : QUEEN;
    Move m(from, to, from_piece, Move::Flags::NONE, NO_PIECE, prom);
    // Seul flag que init_move_flags ne sait pas redériver : le roque (roi
    // se déplaçant de deux colonnes -- aucun coup de roi normal ne le fait).
    // Une entrée corrompue/étrangère qui décoderait ici un faux roque sera
    // rejetée par le check de pseudo-légalité du MovePicker, comme avant.
    if (from_piece == KING && (from - to == 2 || to - from == 2))
        m.set_flags(to > from ? Move::Flags::KING_CASTLE : Move::Flags::QUEEN_CASTLE);
    MoveGen::init_move_flags(board, m);
    return m;
}

struct TTEntry
{
    uint64_t key;
    uint64_t data;
    // Layout de d_pack (62 bits utilisés) :
    // [0..13] move compressé, [14..29] éval statique int16 (TT_NO_EVAL si
    // absente), [30..45] score int16, [46..53] depth, [54..61] flag+age.
    void save(uint64_t k, uint16_t m, int16_t s, int16_t ev, std::uint8_t d, std::uint8_t f)
    {
        uint64_t d_pack = (uint64_t)m |
                          ((uint64_t)(uint16_t)ev << 14) |
                          ((uint64_t)(uint16_t)s << 30) |
                          ((uint64_t)d << 46) |
                          ((uint64_t)f << 54);

        data = d_pack ^ k; // XOR Trick
        key = k;
    }

    bool load(uint64_t k_target, uint16_t &m, int16_t &s, int16_t &ev, std::uint8_t &d, std::uint8_t &f) const
    {
        const uint64_t k = key;

        if (k == k_target)
        {
            const uint64_t d_xor = data;
            const uint64_t d_pack = d_xor ^ k;
            m = static_cast<uint16_t>(d_pack & 0x3FFF);
            ev = static_cast<int16_t>((d_pack >> 14) & 0xFFFF);
            s = static_cast<int16_t>((d_pack >> 30) & 0xFFFF);
            d = static_cast<std::uint8_t>((d_pack >> 46) & 0xFF);
            f = static_cast<std::uint8_t>((d_pack >> 54) & 0xFF);
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
        // Un probe TT = un accès aléatoire dans des centaines de MB : chaque
        // probe paie sinon un dTLB miss en plus du cache miss.
        cpu::advise_huge_pages(table.get(), n * sizeof(TTBucket));
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

    void store(uint64_t key, int depth, int ply, int score, std::uint8_t flag, Move move, int eval = TT_NO_EVAL)
    {
        TTBucket &bucket = table[key & index_mask];
        int replace_idx = -1;
        int min_priority = 1000000;
        const uint16_t m_new = compress_tt_move(move);
        const int16_t ev_new = static_cast<int16_t>(eval);

        for (int i = 0; i < 4; ++i)
        {
            // Lecture directe non-atomique
            uint64_t k_old = bucket.entries[i].key;

            if (k_old == key)
            {
                uint16_t m_prev;
                int16_t s_prev;
                int16_t ev_prev;
                std::uint8_t d_prev;
                std::uint8_t f_prev;
                if (bucket.entries[i].load(key, m_prev, s_prev, ev_prev, d_prev, f_prev))
                {
                    bool old_gen = ((f_prev ^ current_age) & 0xFC) != 0;
                    bool new_is_mate = abs(score) > engine_constants::eval::MateScore - 256;
                    bool old_is_mate = abs(s_prev) > engine_constants::eval::MateScore - 256;

                    bool replace;

                    if (new_is_mate && old_is_mate)
                        replace = abs(score) < abs(s_prev); // PLUS COURT = MEILLEUR
                    else
                        replace = depth >= d_prev || old_gen;

                    // L'éval statique ne périme jamais (fonction de la seule
                    // position) : un store qui n'en apporte pas hérite de
                    // celle déjà en place au lieu de l'effacer.
                    const int16_t ev_merged = (ev_new != TT_NO_EVAL) ? ev_new : ev_prev;
                    if (replace)
                    {
                        bucket.entries[i].save(key, (move != 0) ? m_new : m_prev,
                                               (int16_t)score_to_tt(score, ply), ev_merged, (std::uint8_t)depth, flag | current_age);
                    }
                    else if (ev_merged != ev_prev)
                    {
                        // Entrée conservée (plus profonde) mais on densifie :
                        // on lui greffe l'éval fraîchement calculée.
                        bucket.entries[i].save(key, m_prev, s_prev, ev_merged, d_prev, f_prev);
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
            std::uint8_t d_old = (std::uint8_t)((d_pack >> 46) & 0xFF);
            std::uint8_t f_old = (std::uint8_t)((d_pack >> 54) & 0xFF);

            int priority = d_old + (((f_old ^ current_age) & 0xFC) ? 0 : 100);
            if (priority < min_priority)
            {
                min_priority = priority;
                replace_idx = i;
            }
        }

        if (replace_idx != -1)
            bucket.entries[replace_idx].save(key, m_new, (int16_t)score_to_tt(score, ply), ev_new, (std::uint8_t)depth, flag | current_age);
    }

    // `board` doit être la position correspondant à `key` : elle sert à
    // reconstruire le Move complet (voir decompress_tt_move). `tt_eval`
    // reçoit l'éval statique de l'entrée (TT_NO_EVAL si absente/pas
    // d'entrée), même quand le probe ne produit pas de cutoff.
    bool probe(const Board &board, uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move, TTFlag &flag, int &tt_eval)
    {
        TTBucket &bucket = table[key & index_mask];
        bool found_move = false;
        tt_eval = TT_NO_EVAL;

        for (int i = 0; i < 4; ++i)
        {
            uint16_t m;
            int16_t s;
            int16_t ev;
            std::uint8_t d;
            std::uint8_t f;
            if (!bucket.entries[i].load(key, m, s, ev, d, f))
                continue;

            if (!found_move)
            {
                best_move = decompress_tt_move(board, m);
                tt_eval = ev;
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

    Move get_move(const Board &board) const
    {
        const uint64_t key = board.get_hash();
        TTBucket &bucket = table[key & index_mask];
        for (int i = 0; i < 4; ++i)
        {
            uint16_t m;
            int16_t s;
            int16_t ev;
            std::uint8_t d;
            std::uint8_t f;
            if (bucket.entries[i].load(key, m, s, ev, d, f))
                return decompress_tt_move(board, m);
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