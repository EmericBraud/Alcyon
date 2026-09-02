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

    void next_generation() { current_age_ = static_cast<std::uint8_t>(current_age_ + 1); }

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
        const std::uint64_t packed = pack(key, score_cp, depth, current_age_);
        // Store data before key so a torn read (data half-written) is
        // caught by the XOR check in probe(), same trick as TTEntry.
        slot.data.store(packed, std::memory_order_relaxed);
        slot.key.store(key ^ packed, std::memory_order_release);
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
