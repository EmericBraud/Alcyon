#pragma once

// This header is meant to import important libraries only once.
#include <array>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <utility>
#include <climits>
#include <random>
#include <chrono>
#include <bit>
#include <fstream>
#include <memory>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <thread>
#include <expected>
#include <print>
#include <deque>
#include <optional>
#include <experimental/scope>
#include <cstddef>

#ifdef __BMI2__
#include <immintrin.h>
#endif
#ifdef __SSE2__
#include <xmmintrin.h>
#endif
#define BOARD_SIZE 64
#define STARTING_POS_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#define EN_PASSANT_SQ_NONE 255
#define N_PIECES_TYPE 12
#define N_PIECES_TYPE_HALF 6
#define FEN_INIT_POS "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#define FEN_HARD_PROBLEM_1 "8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1" // Successfully solved after depth 20 (20s/move)

constexpr int MATE_SCORE = 10000;
