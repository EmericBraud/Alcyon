#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>

#include "gpu_config.hpp"

// Small, separate TT holding GPU-computed scores, keyed by zobrist hash --
// see docs/gpu-async-eval/architecture.md. Deliberately NOT merged into
// TTEntry/TranspositionTable (transp_table.hpp): different writer (the one
// dedicated GPU-prep thread, not search workers) and different lifetime
// semantics (a GPU score is a hint the main TT probe consults, not a
// search result with alpha/beta bounds).
//
// Same lockless XOR-packed validation idiom as TTEntry (transp_table.hpp)
// -- one thread ever calls store() (the GPU-prep thread), many threads
// call probe() concurrently (every search worker's TT probe), so this
// needs to be safe against a reader observing a half-written entry
// without any lock.
namespace gpu_eval {

class GpuTT {
public:
    GpuTT() { resize(kGpuTTSizeBytes); }

    void resize(std::size_t size_bytes) {
        std::size_t n = 1;
        while (n * 2 * sizeof(Entry) <= size_bytes) {
            n *= 2;
        }
        capacity_ = n;
        index_mask_ = n - 1;
        table_ = std::make_unique<Entry[]>(n);
    }

    void clear() {
        for (std::size_t i = 0; i < capacity_; ++i) {
            table_[i].key.store(0, std::memory_order_relaxed);
            table_[i].data.store(0, std::memory_order_relaxed);
        }
    }

    // Resets the "how many computed positions actually got used" counters
    // (see stores()/useful_hits()/usage_ratio_percent() below) alongside
    // the generation bump, so they're scoped to "since the last `go`",
    // matching what that ratio is actually meant to answer.
    void next_generation() {
        current_age_ = static_cast<std::uint8_t>(current_age_ + 1);
        stores_.store(0, std::memory_order_relaxed);
        useful_hits_.store(0, std::memory_order_relaxed);
        agreements_.store(0, std::memory_order_relaxed);
        disagreements_.store(0, std::memory_order_relaxed);
        cnn_minus_nnue_total_.store(0, std::memory_order_relaxed);
        cnn_total_.store(0, std::memory_order_relaxed);
        nnue_total_.store(0, std::memory_order_relaxed);
        cnn_sq_total_.store(0, std::memory_order_relaxed);
        nnue_sq_total_.store(0, std::memory_order_relaxed);
        cnn_nnue_total_.store(0, std::memory_order_relaxed);
        cnn_vs_nnue_samples_.store(0, std::memory_order_relaxed);
        redundant_stores_.store(0, std::memory_order_relaxed);
        collisions_.store(0, std::memory_order_relaxed);
        flips_.store(0, std::memory_order_relaxed);
        flip_samples_.store(0, std::memory_order_relaxed);
        busy_ns_.store(0, std::memory_order_relaxed);
        idle_ns_.store(0, std::memory_order_relaxed);
    }

    // For freshness checks at the call site (see transp_table.hpp's
    // probe()) -- a GPU score computed for a previous search root (an
    // earlier move played, or an earlier `go`) is stale and must not be
    // trusted, even if the zobrist key happens to still match (possible
    // via transposition into a position last visited several moves ago).
    std::uint8_t current_age() const { return current_age_; }

    // depth: the search depth (plies) at which this position was captured
    // as a PV leaf -- used only as a freshness/replacement hint, no bound
    // semantics (this is a static eval cache, not a search result).
    void store(std::uint64_t key, std::int16_t score_cp, std::uint8_t depth) {
        Entry &slot = table_[key & index_mask_];

        // Measurement only (single-writer thread -- the GPU-prep thread
        // is the only caller of store(), so this read-before-write is
        // race-free): was this slot already occupied, and if so, by the
        // SAME position (a redundant recompute -- the GPU thread just
        // redid work for a position it had already scored) or a
        // DIFFERENT one (a real hash-slot collision, aliased out)?
        const std::uint64_t existing_key_xor = slot.key.load(std::memory_order_relaxed);
        const std::uint64_t existing_data = slot.data.load(std::memory_order_relaxed);
        if (existing_key_xor != 0 || existing_data != 0) {
            if ((existing_key_xor ^ existing_data) == key) {
                redundant_stores_.fetch_add(1, std::memory_order_relaxed);
            } else {
                collisions_.fetch_add(1, std::memory_order_relaxed);
            }
        }

        const std::uint64_t packed = pack(key, score_cp, depth, current_age_);
        // Store data before key so a torn read (data half-written) is
        // caught by the XOR check in probe(), same trick as TTEntry.
        slot.data.store(packed, std::memory_order_relaxed);
        slot.key.store(key ^ packed, std::memory_order_release);
        stores_.fetch_add(1, std::memory_order_relaxed);
    }

    bool probe(std::uint64_t key, std::int16_t &out_score, std::uint8_t &out_depth, std::uint8_t &out_age) const {
        const Entry &slot = table_[key & index_mask_];
        const std::uint64_t stored_key_xor = slot.key.load(std::memory_order_acquire);
        const std::uint64_t packed = slot.data.load(std::memory_order_relaxed);
        if ((stored_key_xor ^ packed) != key) {
            return false;
        }
        out_score = static_cast<std::int16_t>(packed & 0xFFFF);
        out_depth = static_cast<std::uint8_t>((packed >> 16) & 0xFF);
        out_age = static_cast<std::uint8_t>((packed >> 24) & 0xFF);
        return true;
    }

    // Call from transp_table.hpp's probe() whenever a fresh (age-matching)
    // gpu_tt hit actually gets used to produce a returned score -- see
    // usage_ratio_percent()'s doc below for what this measures.
    void record_useful_hit() { useful_hits_.fetch_add(1, std::memory_order_relaxed); }

    // Call from GpuQueue::run() (the GPU-prep thread itself) once per
    // freshly-inferred position, with the CNN's score and NNUE's eval of
    // the SAME (settled) position. Pure measurement -- nothing acts on
    // the result any more (a bounded resolution search used to; see the
    // note at the top of gpu_queue.cpp for why it was removed).
    //
    // Two things are tracked: how OFTEN the two models disagree beyond
    // kNeutralAgreementMaxGapCp, and the SIGNED mean of (cnn - nnue). A
    // large signed mean means the two are simply on different scales --
    // fixable by recalibrating the CNN's output, which would collapse
    // the disagreement rate. A near-zero signed mean alongside a high
    // disagreement rate means they genuinely differ position by
    // position, which is the interesting case.
    void record_cnn_vs_nnue(int cnn_cp, int nnue_cp) {
        if (std::abs(cnn_cp - nnue_cp) <= kNeutralAgreementMaxGapCp) {
            agreements_.fetch_add(1, std::memory_order_relaxed);
        } else {
            disagreements_.fetch_add(1, std::memory_order_relaxed);
        }
        cnn_minus_nnue_total_.fetch_add(cnn_cp - nnue_cp, std::memory_order_relaxed);
        cnn_total_.fetch_add(cnn_cp, std::memory_order_relaxed);
        nnue_total_.fetch_add(nnue_cp, std::memory_order_relaxed);
        cnn_sq_total_.fetch_add(static_cast<std::int64_t>(cnn_cp) * cnn_cp, std::memory_order_relaxed);
        nnue_sq_total_.fetch_add(static_cast<std::int64_t>(nnue_cp) * nnue_cp, std::memory_order_relaxed);
        cnn_nnue_total_.fetch_add(static_cast<std::int64_t>(cnn_cp) * nnue_cp, std::memory_order_relaxed);
        cnn_vs_nnue_samples_.fetch_add(1, std::memory_order_relaxed);
    }

    // Least-squares fit of nnue ~ a + slope*cnn over this search's
    // sampled positions, plus how well the two actually track each other.
    //  - slope far from 1 means the CNN's cp scale is simply wrong (its
    //    raw output is a logit x kScoreScale, see metal_backend.mm) and a
    //    single multiplier fixes it.
    //  - a low correlation means no rescaling will help: the CNN just
    //    isn't ranking these positions the way the search's own eval does.
    double cnn_to_nnue_slope() const {
        const double vc = var_cnn();
        return vc <= 0.0 ? 0.0 : covariance() / vc;
    }
    // Constant part of that fit, in cp. This is the number
    // kCnnToNnueOffsetCp (gpu_config.hpp) exists to cancel, so once that
    // correction is applied this should read near 0 -- it is the
    // self-check on the correction.
    double cnn_to_nnue_intercept_cp() const { return avg_nnue_cp() - cnn_to_nnue_slope() * avg_cnn_cp(); }
    double cnn_nnue_correlation() const {
        const double d = var_cnn() * var_nnue();
        return d <= 0.0 ? 0.0 : covariance() / std::sqrt(d);
    }

    // Wall-clock accounting for GpuQueue::run()'s main loop -- call once
    // per loop iteration with how long the "real work" (drain + quietify
    // + encode + infer_batch + agree-or-resolve + store) took, or how
    // long the idle sleep_for() wait took when the queue was empty.
    // Together these answer "how saturated is the GPU-prep thread" --
    // see gpu_thread_busy_percent().
    // See qsearch.cpp's consumption site: called once per GPU score the
    // search actually read, with whether that score put the node on the
    // other side of beta than the eval it replaced would have. Only
    // populated when measure_decision_flips() is on.
    void record_decision_flip(bool flipped) {
        flip_samples_.fetch_add(1, std::memory_order_relaxed);
        if (flipped) {
            flips_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    double decision_flip_rate_percent() const {
        const std::uint64_t n = flip_samples_.load(std::memory_order_relaxed);
        return n == 0 ? 0.0 : (100.0 * static_cast<double>(flips_.load(std::memory_order_relaxed)) / static_cast<double>(n));
    }

    void record_busy_ns(std::uint64_t ns) { busy_ns_.fetch_add(ns, std::memory_order_relaxed); }
    void record_idle_ns(std::uint64_t ns) { idle_ns_.fetch_add(ns, std::memory_order_relaxed); }

    double gpu_thread_busy_percent() const {
        const std::uint64_t busy = busy_ns_.load(std::memory_order_relaxed);
        const std::uint64_t idle = idle_ns_.load(std::memory_order_relaxed);
        const std::uint64_t total = busy + idle;
        return total == 0 ? 0.0 : (100.0 * static_cast<double>(busy) / static_cast<double>(total));
    }

    double avg_cnn_minus_nnue_cp() const { return mean_of(cnn_minus_nnue_total_); }
    // Read alongside avg_cnn_minus_nnue_cp() to tell a constant OFFSET
    // (avg_cnn - avg_nnue large, ratio near 1) from a SCALE mismatch
    // (avg_cnn / avg_nnue far from 1) -- they need different fixes.
    double avg_cnn_cp() const { return mean_of(cnn_total_); }
    double avg_nnue_cp() const { return mean_of(nnue_total_); }

    std::uint64_t stores() const { return stores_.load(std::memory_order_relaxed); }
    std::uint64_t useful_hits() const { return useful_hits_.load(std::memory_order_relaxed); }
    std::uint64_t agreements() const { return agreements_.load(std::memory_order_relaxed); }
    std::uint64_t disagreements() const { return disagreements_.load(std::memory_order_relaxed); }
    std::uint64_t redundant_stores() const { return redundant_stores_.load(std::memory_order_relaxed); }
    std::uint64_t collisions() const { return collisions_.load(std::memory_order_relaxed); }

    // How often the CNN and NNUE land more than kNeutralAgreementMaxGapCp
    // apart -- read together with avg_cnn_minus_nnue_cp() (see
    // record_cnn_vs_nnue).
    double disagreement_rate_percent() const {
        const std::uint64_t n = cnn_vs_nnue_samples_.load(std::memory_order_relaxed);
        return n == 0 ? 0.0 : (100.0 * static_cast<double>(disagreements()) / static_cast<double>(n));
    }

    // Fraction of stores() that overwrote a DIFFERENT position's slot
    // (real hash aliasing, capacity pressure) vs. a matching one
    // (recomputed the same position -- see redundant_stores()).
    double collision_rate_percent() const {
        const std::uint64_t s = stores();
        return s == 0 ? 0.0 : (100.0 * static_cast<double>(collisions()) / static_cast<double>(s));
    }
    double redundant_rate_percent() const {
        const std::uint64_t s = stores();
        return s == 0 ? 0.0 : (100.0 * static_cast<double>(redundant_stores()) / static_cast<double>(s));
    }

    std::size_t capacity() const { return capacity_; }

    // Of the positions the GPU thread computed and stored (this search),
    // what fraction were ever actually consulted by the search and used
    // to produce a result (negamax.cpp's should_qsearch branch). THE
    // number to watch: it is what says whether this subsystem is visible
    // to the search at all. NOT a cache hit rate -- a single computed
    // position can be probed (and used) many times by different
    // workers/nodes, so this can exceed 100%.
    double usage_ratio_percent() const {
        const std::uint64_t s = stores();
        return s == 0 ? 0.0 : (100.0 * static_cast<double>(useful_hits()) / static_cast<double>(s));
    }

private:
    struct Entry {
        std::atomic<std::uint64_t> key{0};
        std::atomic<std::uint64_t> data{0};
    };

    static std::uint64_t pack(std::uint64_t key, std::int16_t score_cp, std::uint8_t depth, std::uint8_t age) {
        (void)key;
        return (static_cast<std::uint64_t>(static_cast<std::uint16_t>(score_cp))) |
               (static_cast<std::uint64_t>(depth) << 16) |
               (static_cast<std::uint64_t>(age) << 24);
    }

    std::unique_ptr<Entry[]> table_;
    std::size_t capacity_ = 0;
    std::size_t index_mask_ = 0;
    std::uint8_t current_age_ = 0;
    std::atomic<std::uint64_t> stores_{0};
    std::atomic<std::uint64_t> useful_hits_{0};
    std::atomic<std::uint64_t> agreements_{0};
    std::atomic<std::uint64_t> disagreements_{0};
    std::atomic<std::int64_t> cnn_minus_nnue_total_{0};
    std::atomic<std::int64_t> cnn_total_{0};
    std::atomic<std::int64_t> nnue_total_{0};
    std::atomic<std::int64_t> cnn_sq_total_{0};
    std::atomic<std::int64_t> nnue_sq_total_{0};
    std::atomic<std::int64_t> cnn_nnue_total_{0};
    std::atomic<std::uint64_t> cnn_vs_nnue_samples_{0};

    double mean_of(const std::atomic<std::int64_t> &total) const {
        const std::uint64_t n = cnn_vs_nnue_samples_.load(std::memory_order_relaxed);
        return n == 0 ? 0.0 : static_cast<double>(total.load(std::memory_order_relaxed)) / static_cast<double>(n);
    }
    double covariance() const { return mean_of(cnn_nnue_total_) - avg_cnn_cp() * avg_nnue_cp(); }
    double var_cnn() const { return mean_of(cnn_sq_total_) - avg_cnn_cp() * avg_cnn_cp(); }
    double var_nnue() const { return mean_of(nnue_sq_total_) - avg_nnue_cp() * avg_nnue_cp(); }
    std::atomic<std::uint64_t> flips_{0};
    std::atomic<std::uint64_t> flip_samples_{0};
    std::atomic<std::uint64_t> busy_ns_{0};
    std::atomic<std::uint64_t> idle_ns_{0};
    std::atomic<std::uint64_t> redundant_stores_{0};
    std::atomic<std::uint64_t> collisions_{0};
};

// One shared instance, mirroring TranspositionTable's ownership pattern
// (owned by value in EngineManager, handed to workers by reference) --
// this one is simple/small enough to own as a singleton instead, since
// unlike the main TT it isn't resized via a UCI option.
inline GpuTT &shared_gpu_tt() {
    static GpuTT instance;
    return instance;
}

} // namespace gpu_eval
