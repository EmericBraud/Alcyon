#pragma once

#include <array>

#include "common/constants.hpp"
#include "engine/config/config.hpp"

#ifdef TEXEL_TUNING
#include "engine/config/texel_param.hpp"

#define CONSTEXPR_GUARD
#define CONST_GUARD
#define TUNABLE_TYPE TunableParam
#define EVAL_ACCUM_TYPE double
#else
#define EVAL_ACCUM_TYPE int
#define CONSTEXPR_GUARD constexpr
#define CONST_GUARD const
#define TUNABLE_TYPE int
#endif

namespace engine_constants::eval
{
    //clang-format off
    static constexpr int MvvLvaTable[7][7] = {
        // Attaquants:  P   N    B    R    Q    K  NONE
        /* PAWN   */ {105, 104, 103, 102, 101, 100, 0},
        /* KNIGHT */ {205, 204, 203, 202, 201, 200, 0},
        /* BISHOP */ {305, 304, 303, 302, 301, 300, 0},
        /* ROOK   */ {405, 404, 403, 402, 401, 400, 0},
        /* QUEEN  */ {505, 504, 503, 502, 501, 500, 0},
        /* KING   */ {605, 604, 603, 602, 601, 600, 0},
        /* NONE   */ {0, 0, 0, 0, 0, 0, 0}};
    //clang-format on

    // ================ PST ================

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        -16, 0, -3, 2, 15, 33, 49, -5,
        -18, -17, 12, 5, 23, 6, 24, -9,
        -30, -22, 4, 21, 15, 6, -19, -52,
        -11, -2, 4, 14, 28, 9, -5, -31,
        -12, 0, 30, 10, 33, 79, 6, -12,
        11, 0, -3, 7, 10, 6, -11, -21,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_knight_table[constants::BoardSize] = {
        -30, 24, -17, 9, 12, 18, 17, -37,
        2, -15, 22, 49, 48, 34, 13, 27,
        4, 26, 51, 48, 54, 53, 50, 2,
        23, 10, 56, 48, 64, 69, 31, 6,
        18, 59, 42, 90, 62, 71, 50, 30,
        -17, 6, 31, 58, 61, 69, 31, -6,
        -69, -37, 47, 3, 32, 42, -14, -11,
        -119, -22, -23, -19, -10, -40, -22, -54
    };
    static CONSTEXPR_GUARD TUNABLE_TYPE mg_bishop_table[constants::BoardSize] = {
        11, 0, 29, 12, 6, 6, -3, 14,
        -9, 52, 29, 26, 33, 45, 69, 16,
        33, 36, 39, 24, 34, 42, 22, 28,
        -8, 4, 20, 38, 45, 15, 11, -19,
        -26, -11, 8, 30, 26, -6, 7, -1,
        -9, 4, 28, 6, 27, 23, 23, 32,
        -36, -15, -20, -11, -4, 18, -15, 4,
        -25, -26, -14, -24, -24, -21, -9, -30
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_rook_table[constants::BoardSize] = {
        -14, -8, 13, 19, 25, 16, -23, 5,
        -40, -18, -18, -7, 3, 4, 10, -42,
        -43, -25, -16, 1, -2, 1, 4, -4,
        -44, -28, -36, -22, -15, -8, 0, -22,
        -39, -16, -8, 0, -19, 1, 4, 6,
        -5, -12, -26, 5, 10, 16, 10, 8,
        -14, -40, 6, 22, 6, 33, 31, 16,
        2, 9, 5, 9, 11, 11, 9, 1
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_queen_table[constants::BoardSize] = {
        11, 5, 16, 30, 15, -11, -18, -17,
        -4, 10, 10, 20, 25, 25, 20, -2,
        -17, 4, -5, -1, 2, 10, 11, -6,
        -20, -13, -21, -18, -7, 4, 3, -8,
        -34, -25, -22, -32, -14, -4, -5, 16,
        -29, -20, -21, 11, 12, 31, 41, 55,
        -31, -51, -26, -15, -17, 42, 9, 47,
        -30, -13, -3, 9, 10, 12, -10, -8
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_king_table[constants::BoardSize] = {
        -5, 48, 35, -34, 30, -13, 61, 41,
        14, 6, -2, -40, -39, -13, 32, 27,
        -8, 17, -23, -31, -42, -31, -1, -27,
        -12, -8, -14, -27, -24, -33, -17, -32,
        -20, -17, -14, -29, -33, -24, -18, -24,
        -17, -4, -21, -26, -33, -15, -9, -16,
        -17, -13, -22, -35, -35, -17, -19, -17,
        -17, -21, -29, -38, -37, -27, -22, -19
    };
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_king_table[constants::BoardSize] = {
        -50, -44, -18, -7, -25, -19, -58, -91,
        -40, -13, 12, 28, 27, 16, -11, -40,
        -40, -7, 17, 30, 34, 24, 3, -21,
        -45, -8, 16, 28, 32, 29, 9, -19,
        -38, 4, 16, 27, 26, 33, 26, -5,
        -27, 20, 19, 21, 25, 39, 44, 1,
        -26, 9, 6, 7, 12, 30, 21, -2,
        -35, -18, -15, -8, -7, -5, -11, -30
    };
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        33, 24, 28, 22, 21, 22, 7, 5,
        25, 25, 12, 14, 18, 18, 11, 9,
        38, 34, 11, 4, 9, 9, 20, 19,
        51, 43, 21, -1, 4, 11, 29, 30,
        78, 56, 36, -4, -15, 0, 29, 43,
        75, 73, 43, 16, 6, 10, 51, 49,
        0, 0, 0, 0, 0, 0, 0, 0
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_knight_table[constants::BoardSize] = {
        -39, -37, -20, -8, -25, -15, -43, -30,
        -25, -25, -31, -30, -32, -16, -22, -28,
        -31, -19, -29, 1, 1, -25, -34, -36,
        -23, -4, 4, 9, 16, -7, -11, -14,
        -21, -4, 7, 17, 16, 14, -2, -20,
        -36, -31, 2, 6, -14, -1, -24, -28,
        -39, -29, -25, -15, -28, -32, -35, -37,
        -67, -42, -31, -28, -29, -42, -30, -66
    };
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_bishop_table[constants::BoardSize] = {
        -18, -9, -21, -4, -6, -3, -22, -15,
        -10, -27, -20, -12, -8, -11, -20, -20,
        -22, -1, -4, 9, 6, -9, -18, -23,
        -9, -7, 2, -2, -2, 3, -17, -11,
        1, 8, -2, 7, 4, -9, -11, -12,
        -5, -1, -10, 2, -11, 7, -2, -10,
        -21, -6, -8, -19, -7, -15, -10, -36,
        -16, -31, -24, -12, -18, -12, -16, -15
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_rook_table[constants::BoardSize] = {
        16, 14, 9, 7, -3, 5, 8, -20,
        7, 5, 17, 12, 3, 2, 1, -3,
        7, 8, 2, 4, 4, -2, -8, -10,
        18, 14, 18, 17, 14, 8, 2, -4,
        20, 14, 14, 8, 10, 9, 4, 7,
        18, 18, 15, 5, 3, 7, 12, 5,
        24, 39, 23, 20, 10, 13, 15, 9,
        35, 22, 26, 16, 12, 14, 14, 18
    };
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_queen_table[constants::BoardSize] = {
        -15, -23, -23, -48, -10, -16, -17, -21,
        -13, -8, -10, -10, -19, -14, -14, -8,
        -25, -31, 14, 2, 9, 12, 13, -10,
        -6, 0, 9, 34, 33, 14, 12, 0,
        -18, 5, 1, 35, 43, 35, 18, 11,
        -24, -9, 14, 8, 35, 21, 19, 0,
        -11, 1, 1, 18, 21, 41, 14, -3,
        -16, -6, 2, 15, 14, 13, -8, 6
    };

    // Table de correspondance pour le milieu de jeu
    static CONST_GUARD TUNABLE_TYPE *mg_tables[] = {
        mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table};

    static CONST_GUARD TUNABLE_TYPE *eg_tables[] = {
        eg_pawn_table, eg_knight_table, eg_bishop_table, eg_rook_table, eg_queen_table, eg_king_table};

    static const int pawnPhase = 0;
    static const int knightPhase = 1;
    static const int bishopPhase = 1;
    static const int rookPhase = 2;
    static const int queenPhase = 4;
    static const int totalPhase = 24;

    static constexpr int alphaBetaMargin = 110;

    static const int phase_values[] = {pawnPhase, knightPhase, bishopPhase, rookPhase, queenPhase, totalPhase};
    // ================ Mobility ================
    static CONSTEXPR_GUARD TUNABLE_TYPE knight_mob[] = {
        -31, 11, 26, 30, 39, 41, 43, 45,
        44
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE bishop_mob[] = {
        -17, -6, 6, 11, 18, 25, 29, 34,
        39, 39, 46, 45, 47, 31
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE rook_mob[] = {
        7, 20, 23, 30, 30, 39, 44, 51,
        57, 60, 66, 72, 81, 83, 81
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE queen_mob[] = {
        33, 35, 36, 36, 39, 46, 50, 50,
        51, 59, 64, 67, 68, 73, 76, 78,
        81, 82, 90, 91, 92, 99, 99, 99,
        88, 101, 85, 82
    };

    // ================ Material ================
    static CONSTEXPR_GUARD std::array<TUNABLE_TYPE, constants::PieceTypeCount> pieces_score = {
        94, 367, 393, 604, 1220, 10000
    };

    // ================ Pawn structure ================
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesMgMalus = 0;
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesEgMalus = -27;

    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesMgMalus = -26;
    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesEgMalus = -6;

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_mg[] = {
        0, 5, 1, -10, 4, -11, 25, 0
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_eg[] = {
        0, 4, 9, 34, 65, 149, 216, 0
    };

    // ================ King safety ================
    static CONSTEXPR_GUARD TUNABLE_TYPE openFileMalus = -52;
    static CONSTEXPR_GUARD TUNABLE_TYPE semiOpenFileMalus = -19;

    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesOpenFileMalus = -11;
    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesSemiOpenFileMalus = -3;

    // ================ Structured threats ================
    static CONSTEXPR_GUARD TUNABLE_TYPE defendedThreatsBonus[constants::PieceTypeCount][constants::PieceTypeCount] = {
        {0, 0, 0, 0, 0, 0},
        {-8, 4, 23, 37, 3, 57},
        {-8, 14, 20, 13, 14, 55},
        {-2, 8, 15, 21, 23, 33},
        {-3, -8, -4, -7, 30, 74},
        {0, 0, 0, 0, 0, 0}
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE undefendedThreatsBonus[constants::PieceTypeCount][constants::PieceTypeCount] = {
        {0, 0, 0, 0, 0, 0},
        {8, -6, 24, 26, 5, -8},
        {5, 29, -7, 13, 30, 24},
        {14, 24, 15, -2, 17, -2},
        {-2, 12, 17, 15, -2, 65},
        {0, 0, 0, 0, 0, 0}
    };

    static CONSTEXPR_GUARD TUNABLE_TYPE pawnPushThreatBonus = 16;

    // ================ Bishop pair ================
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairMgBonus = 62;
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairEgBonus = 39;

    // ================ Endgame / Mop-up ================
    static CONSTEXPR_GUARD TUNABLE_TYPE kingDistFromCenterBonus = 26;
    static CONSTEXPR_GUARD TUNABLE_TYPE closeKingBonus = -19;

    static CONSTEXPR_GUARD TUNABLE_TYPE maxDistBetweenKings = 11;

}