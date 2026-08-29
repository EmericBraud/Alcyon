#pragma once

#include <cstdint>
#include <string>

// Isolation boundary: chess26's own engine headers (Board, MoveGen, etc.,
// unqualified/global names) and the vendored nnue-pytorch's chess.h
// (namespace `chess`) both exist in this codebase and would collide if
// ever included in the same translation unit alongside `using chess::...`
// declarations (as plane_batch.cpp has). This header only exposes a plain
// std::string/int32_t function signature, so callers never need to
// #include the engine's headers themselves — keep the two ecosystems in
// separate .cpp files (see nnue_bridge.cpp).

namespace chess26::cnn {

// Computes chess26's own NNUE static evaluation for the position given by
// `fen`, from the side-to-move's perspective (positive = side to move is
// better) — same convention as binpack::TrainingDataEntry::score/result
// used elsewhere in this loader. Not incremental: rebuilds both
// perspectives' accumulators from scratch for this one position.
//
// `nnue_path` selects the .nnue weight file; weights are cached
// process-wide (see nnue_eval.hpp's shared_base_model), so repeated calls
// with the same path only pay the deserialization cost once. Thread-safe.
std::int32_t compute_nnue_score(const std::string& fen, const std::string& nnue_path);

}  // namespace chess26::cnn
