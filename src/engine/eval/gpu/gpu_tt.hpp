#pragma once

#include <atomic>
#include <cstdint>
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
        disagreement_hits_.store(0, std::memory_order_relaxed);
        redundant_stores_.store(0, std::memory_order_relaxed);
        collisions_.store(0, std::memory_order_relaxed);
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

    // Call whenever a fresh gpu_tt hit is found but NNUE and the GPU
    // score disagree on the cutoff decision (see transp_table.hpp's
    // classify_cut_decision-based check) -- the override is skipped and
    // a real search happens instead, but this is NOT wasted work: the
    // disagreement itself is the value here (it caught a contested
    // position and avoided trusting either model's unverified score).
    // Separate from useful_hits() (a trusted shortcut) since they're
    // both "useful" in different ways -- only a stored position that's
    // NEVER hit again by any probe() is truly wasted GPU-thread work.
    void record_disagreement_hit() { disagreement_hits_.fetch_add(1, std::memory_order_relaxed); }

    std::uint64_t stores() const { return stores_.load(std::memory_order_relaxed); }
    std::uint64_t useful_hits() const { return useful_hits_.load(std::memory_order_relaxed); }
    std::uint64_t disagreement_hits() const { return disagreement_hits_.load(std::memory_order_relaxed); }
    std::uint64_t redundant_stores() const { return redundant_stores_.load(std::memory_order_relaxed); }
    std::uint64_t collisions() const { return collisions_.load(std::memory_order_relaxed); }

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
    // what fraction were TRUSTED -- NNUE and the GPU score agreed on the
    // cutoff decision, so the override actually shortcut a real search
    // (see transp_table.hpp's classify_cut_decision-based check). This is
    // narrower than "was the store useful at all" (see
    // total_value_ratio_percent() below, which also counts disagreement
    // hits): a disagreement is real value too, just not a shortcut. NOT
    // a cache hit rate: a single computed position could be probed (and
    // trusted) many times by different workers/nodes, so this can
    // exceed 100%.
    double usage_ratio_percent() const {
        const std::uint64_t s = stores();
        return s == 0 ? 0.0 : (100.0 * static_cast<double>(useful_hits()) / static_cast<double>(s));
    }

    // Broader than usage_ratio_percent(): counts BOTH trusted overrides
    // (useful_hits) AND disagreements (disagreement_hits) as value
    // extracted from a stored position -- a disagreement still did
    // something useful (flagged a contested position, triggered a real
    // search instead of trusting an unverified score), it just isn't a
    // compute-saving shortcut. Only a store that's never hit again by
    // any probe() at all is genuinely wasted GPU-thread work.
    double total_value_ratio_percent() const {
        const std::uint64_t s = stores();
        return s == 0 ? 0.0 : (100.0 * static_cast<double>(useful_hits() + disagreement_hits()) / static_cast<double>(s));
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
    std::atomic<std::uint64_t> disagreement_hits_{0};
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
