#include "nnue_bridge.h"

#include <memory>
#include <mutex>

#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/eval/nnue/nnue_eval.hpp"

namespace chess26::cnn {

namespace {

// MoveGen's attack tables are process-global mutable state, populated
// once and read-only afterward — safe to share across the loader's
// worker threads once initialized, but must run exactly once before any
// NNUE eval (Full_Threats features call into these tables). Mirrors what
// UCI's constructor does at startup for the real engine binary.
std::once_flag g_movegen_init_flag;
void ensure_movegen_initialized() {
    std::call_once(g_movegen_init_flag, [] { MoveGen::initialize_bitboard_tables(); });
}

}  // namespace

std::int32_t compute_nnue_score(const std::string& fen, const std::string& nnue_path) {
    ensure_movegen_initialized();

    // One NnueEval per thread, lazily (re)built if the requested path
    // changes (never happens in practice — one model per training run —
    // but cheap to handle correctly). The underlying weight tables are
    // shared_ptr-backed and cached process-wide (shared_base_model), so
    // constructing a new NnueEval here is cheap after the first load: no
    // weight deserialization, just a small accumulator-sized copy.
    thread_local std::string cached_path;
    thread_local std::unique_ptr<nnue::NnueEval> cached_eval;
    if (!cached_eval || cached_path != nnue_path) {
        cached_eval = std::make_unique<nnue::NnueEval>(nnue_path);
        cached_path = nnue_path;
    }

    Board board;
    board.load_fen(fen);
    cached_eval->initialize(board);
    return cached_eval->evaluate_abs(board);
}

}  // namespace chess26::cnn
