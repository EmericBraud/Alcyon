#pragma once
#include <cstdint>
#include <cstdlib>
#include <memory>

#include "common/cpu.hpp"
#include "core/move/move.hpp"
#include "engine/config/config.hpp"
#include "engine/eval/gpu/gpu_tt.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/eval/virtual_board.hpp"

// Consultative use of a GPU-computed score: classify a score against the
// current alpha-beta window (would it cause a fail-high, a fail-low, or
// stay inside the window?) and only trust the GPU override when its
// classification AGREES with the position's own NNUE eval -- see
// docs/gpu-async-eval/v5-hybrid-nnue-cnn.md's "risque: la correction peut
// pousser dans le mauvais sens" section, which explicitly recommended a
// consultative (not absolute) integration instead of blindly substituting
// scores. When the two models disagree on the cutoff decision (one wants
// to cut and the other doesn't, or both want to cut but in opposite
// directions), that disagreement is itself the signal: this position's
// static evaluation is contested, so it's worth spending real search on
// instead of shortcutting with either model's unverified opinion.
enum class TTCutDecision : std::uint8_t
{
    Low,     // score <= alpha: this eval would fail low
    Neutral, // alpha < score < beta: stays inside the window
    High     // score >= beta: this eval would fail high
};

inline TTCutDecision classify_cut_decision(int score, int alpha, int beta)
{
    if (score <= alpha)
        return TTCutDecision::Low;
    if (score >= beta)
        return TTCutDecision::High;
    return TTCutDecision::Neutral;
}

// Outcome of should_trust_gpu_score() below -- split into three cases
// (not just trust/don't) so callers can distinguish "the models
// actively disagree on direction" from "they agree on Neutral but the
// magnitude gate rejected it", the latter being the NEW behavior this
// enum exists to measure the real-world impact of (see
// GpuTT::record_neutral_agreement_rejected() and gpuevalstats).
enum class TTTrustVerdict : std::uint8_t
{
    Trust,                // agree, and (if Neutral) close enough in magnitude
    DirectionDisagreement, // classifications differ outright
    NeutralGapTooLarge,   // both Neutral, but |nnue - gpu| > kNeutralAgreementMaxGapCp
};

// Refines plain classification agreement with an asymmetric rule: a
// CUTOFF (both Low, or both High) only needs the right DIRECTION to be
// correct -- alpha-beta pruning discards the branch either way, so the
// exact magnitude never affects the search result once both models
// agree it should be cut (measured: v3's disagreement magnitude with
// NNUE is a strong predictor of NNUE's real error, AUC 0.84 on cp-scale
// ground truth -- see training/cnn/eval_compare/
// v3_vs_v6_error_prediction.py -- but that precision doesn't matter
// here since the cut happens regardless). A NEUTRAL/NEUTRAL agreement
// is different: neither model wants to cut, so the returned score can
// itself propagate further up the tree (PV, comparisons at the parent)
// -- there, "same direction" isn't enough, the two scores also need to
// be close in magnitude (see kNeutralAgreementMaxGapCp) before trusting
// either one. See docs/gpu-async-eval/consultative-eval-measurements.md
// section 6 for the reasoning and the measurements behind it.
inline TTTrustVerdict should_trust_gpu_score(int nnue_score, int gpu_score, int alpha, int beta)
{
    const TTCutDecision nnue_decision = classify_cut_decision(nnue_score, alpha, beta);
    const TTCutDecision gpu_decision = classify_cut_decision(gpu_score, alpha, beta);
    if (nnue_decision != gpu_decision)
        return TTTrustVerdict::DirectionDisagreement;
    if (nnue_decision == TTCutDecision::Neutral &&
        std::abs(nnue_score - gpu_score) > gpu_eval::kNeutralAgreementMaxGapCp)
        return TTTrustVerdict::NeutralGapTooLarge;
    return TTTrustVerdict::Trust;
}

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
                    bool new_is_mate = abs(score) > engine_constants::eval::MateScore - 256;
                    bool old_is_mate = abs(s_prev) > engine_constants::eval::MateScore - 256;

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

    template <Color Us>
    bool probe(uint64_t key, int depth, int ply, int alpha, int beta, int &return_score, Move &best_move, TTFlag &flag, const VBoard &board)
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

            // GPU-eval score override for a position the main TT ALSO
            // already has an entry for (typically a transposition into a
            // position seen elsewhere in the tree) -- see the independent
            // gpu_tt-only check below the loop for the primary case
            // (a brand-new frontier node with no main-TT entry at all,
            // e.g. a PV-leaf child freshly precomputed by the GPU queue --
            // see docs/gpu-async-eval/architecture.md). Non-blocking read
            // of a small separate score cache; a no-op (miss) whenever
            // gpu_eval is disabled or this position was never submitted.
            int16_t gpu_score;
            std::uint8_t gpu_depth, gpu_age;
            if (gpu_eval::shared_gpu_tt().probe(key, gpu_score, gpu_depth, gpu_age) &&
                gpu_age == gpu_eval::shared_gpu_tt().current_age())
            {
                const int nnue_score = Eval::eval_relative<Us>(board, alpha, beta);
                switch (should_trust_gpu_score(nnue_score, gpu_score, alpha, beta))
                {
                case TTTrustVerdict::Trust:
                    score = gpu_score;
                    gpu_eval::shared_gpu_tt().record_useful_hit();
                    break;
                case TTTrustVerdict::NeutralGapTooLarge:
                    // Both agree "don't cut" but disagree enough in
                    // magnitude to reject trusting either score (see
                    // kNeutralAgreementMaxGapCp's doc) -- tracked
                    // separately from a direction disagreement to
                    // measure how often this specific gate actually
                    // fires (see GpuTT::neutral_agreement_rejected()).
                    gpu_eval::shared_gpu_tt().record_neutral_agreement_rejected();
                    break;
                case TTTrustVerdict::DirectionDisagreement:
                    // Disagreement: leave `score` as the main TT's own
                    // (real-search) value -- the GPU's opinion is
                    // discarded for this probe, not force-applied. Still
                    // real value extracted (see record_disagreement_hit's
                    // doc), not wasted work.
                    gpu_eval::shared_gpu_tt().record_disagreement_hit();
                    break;
                }
            }

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

        // NOTE: the brand-new-node case (no main-TT entry at all, e.g. a
        // PV-leaf child freshly precomputed by the GPU queue) used to be
        // handled here too, with the same trust-or-nothing logic as
        // above. Moved to negamax.cpp's should_qsearch branch instead --
        // that call site can recurse back into negamax() with a bounded
        // extra-depth search on disagreement (see
        // gpu_eval::kDisagreementExtensionPlies), which this method
        // cannot do (a TT probe has no business calling back into the
        // search). See docs/gpu-async-eval/consultative-eval-measurements.md
        // section 6 for why a plain reject-and-drop-to-qsearch measured
        // worse than trusting the (imprecise but free) GPU score there.
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