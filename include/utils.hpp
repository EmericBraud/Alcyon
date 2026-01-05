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
#include <immintrin.h>
#include <xmmintrin.h> //x86 / intel only

#define BOARD_SIZE 64
#define STARTING_POS_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define ROOK_ATTACKS_SIZE 102400
#define BISHOP_ATTACKS_SIZE 5248
#define DATA_PATH "../data/"
#define EN_PASSANT_SQ_NONE 255
#define N_PIECES_TYPE 12
#define N_PIECES_TYPE_HALF 6
#define FEN_INIT_POS "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

#define FEN_HARD_PROBLEM_1 "8/3P3k/n2K3p/2p3n1/1b4N1/2p1p1P1/8/3B4 w - - 0 1" // Successfully solved after depth 20 (20s/move)

constexpr int MATE_SCORE = 10000;

using U64 = std::uint64_t;
using bitboard = std::uint64_t;

namespace masks
{
    static constexpr uint8_t king_vicinity_files[8] = {
        0b00000011, // File A (0) : colonnes A et B
        0b00000111, // File B (1) : colonnes A, B, C
        0b00001110, // File C (2) : colonnes B, C, D
        0b00011100, // File D (3) : colonnes C, D, E
        0b00111000, // File E (4) : colonnes D, E, F
        0b01110000, // File F (5) : colonnes E, F, G
        0b11100000, // File G (6) : colonnes F, G, H
        0b11000000  // File H (7) : colonnes G et H
    };
    static constexpr uint64_t file[8] = {
        0x0101010101010101ULL, // Colonne A
        0x0202020202020202ULL, // Colonne B
        0x0404040404040404ULL, // Colonne C
        0x0808080808080808ULL, // Colonne D
        0x1010101010101010ULL, // Colonne E
        0x2020202020202020ULL, // Colonne F
        0x4040404040404040ULL, // Colonne G
        0x8080808080808080ULL  // Colonne H
    };
}

inline int get_lsb_index(U64 bb)
{
    return __builtin_ctzll(bb);
}

inline int pop_lsb(U64 &bb)
{
    int s = get_lsb_index(bb);
    bb &= bb - 1;
    return s;
}

enum Square : int
{
    a1,
    b1,
    c1,
    d1,
    e1,
    f1,
    g1,
    h1,
    a2,
    b2,
    c2,
    d2,
    e2,
    f2,
    g2,
    h2,
    a3,
    b3,
    c3,
    d3,
    e3,
    f3,
    g3,
    h3,
    a4,
    b4,
    c4,
    d4,
    e4,
    f4,
    g4,
    h4,
    a5,
    b5,
    c5,
    d5,
    e5,
    f5,
    g5,
    h5,
    a6,
    b6,
    c6,
    d6,
    e6,
    f6,
    g6,
    h6,
    a7,
    b7,
    c7,
    d7,
    e7,
    f7,
    g7,
    h7,
    a8,
    b8,
    c8,
    d8,
    e8,
    f8,
    g8,
    h8,
};