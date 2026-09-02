#pragma once

#include "core/board/board.hpp"

// Static Exchange Evaluation, as a free function over any Board (not tied to
// SearchWorker) so it can be reused off the main search thread too -- see
// gpu_queue.cpp's quiescence walk, which needs SEE to judge captures without
// constructing a full SearchWorker (shared TT, thread-local heuristics, ...)
// just to settle a position on the (mostly idle) GPU-prep thread.
template <Color Side>
int compute_see(const Board &board, int sq, Piece target, Piece attacker, int from_sq);
