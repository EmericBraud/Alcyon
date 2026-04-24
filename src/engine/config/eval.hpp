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
        -11, 1, -2, 3, 13, 30, 43, 0,
        -10, -14, 13, 6, 21, 8, 23, -1,
        -23, -14, 5, 21, 18, 8, -7, -38,
        -6, 0, 6, 18, 28, 11, 0, -20,
        -4, 5, 27, 9, 25, 55, 8, -2,
        7, 1, -2, 2, 3, 0, -5, -10,
        0, 0, 0, 0, 0, 0, 0, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_knight_table[constants::BoardSize] = {
        -27, 11, -18, 2, 2, 8, 7, -31,
        -7, -14, 10, 32, 31, 19, 3, 14,
        -7, 10, 32, 31, 35, 33, 31, -9,
        6, 1, 29, 25, 38, 35, 8, -7,
        4, 26, 18, 62, 30, 39, 14, 10,
        -14, 5, 22, 45, 43, 46, 20, -7,
        -48, -26, 41, 5, 24, 28, -8, -13,
        -76, -19, -17, -16, -9, -29, -19, -40};
    static CONSTEXPR_GUARD TUNABLE_TYPE mg_bishop_table[constants::BoardSize] = {
        0, -6, 16, 4, -1, -3, -6, 2,
        -11, 35, 15, 13, 19, 28, 49, 5,
        19, 19, 24, 9, 18, 26, 9, 16,
        -9, -2, 4, 25, 26, 4, 2, -19,
        -22, -6, 1, 18, 18, -6, 5, -9,
        -11, 2, 28, 5, 21, 16, 14, 19,
        -31, -11, -15, -6, -1, 14, -9, 2,
        -22, -19, -13, -18, -17, -17, -9, -21};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_rook_table[constants::BoardSize] = {
        -10, -5, 11, 16, 21, 13, -19, 3,
        -29, -10, -8, -2, 3, 4, 8, -27,
        -30, -15, -10, 1, 0, 1, 1, -4,
        -31, -17, -23, -13, -13, -7, -2, -16,
        -24, -9, -4, 2, -11, -1, 1, 2,
        0, -4, -11, 6, 8, 9, 6, 3,
        -4, -21, 11, 19, 8, 23, 20, 10,
        4, 6, 5, 8, 8, 6, 5, 2};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_queen_table[constants::BoardSize] = {
        8, 1, 11, 23, 10, -9, -14, -17,
        -5, 6, 7, 15, 19, 18, 13, -5,
        -17, 1, -5, -2, 1, 4, 8, -7,
        -19, -10, -18, -18, -6, 1, 1, -8,
        -25, -21, -19, -26, -8, -2, -5, 13,
        -25, -15, -6, 2, 19, 20, 27, 37,
        -28, -49, -19, -8, -11, 30, 9, 26,
        -24, -10, -4, 4, 4, 3, -9, -10};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_king_table[constants::BoardSize] = {
        7, 40, 30, -25, 27, -8, 53, 37,
        14, 14, 3, -29, -29, -5, 30, 25,
        -5, 13, -15, -23, -28, -18, 4, -15,
        -11, -13, -19, -29, -27, -27, -16, -21,
        -19, -21, -23, -34, -37, -27, -20, -20,
        -17, -14, -26, -33, -37, -23, -15, -16,
        -17, -19, -27, -38, -38, -24, -21, -17,
        -17, -23, -31, -40, -39, -30, -23, -18};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_king_table[constants::BoardSize] = {
        -34, -31, -14, -7, -21, -15, -46, -73,
        -28, -11, 8, 20, 20, 11, -8, -31,
        -29, -5, 13, 22, 24, 18, 2, -18,
        -31, -5, 14, 22, 26, 22, 7, -16,
        -26, 3, 14, 22, 21, 26, 20, -4,
        -19, 14, 15, 17, 19, 29, 31, -1,
        -19, 3, 3, 5, 8, 20, 11, -6,
        -25, -15, -11, -6, -6, -7, -11, -23};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        28, 22, 25, 20, 20, 21, 7, 6,
        21, 22, 13, 15, 18, 18, 11, 8,
        31, 28, 11, 6, 10, 10, 17, 18,
        42, 37, 20, 1, 5, 11, 25, 25,
        63, 47, 33, 3, -3, 8, 27, 37,
        60, 56, 35, 19, 15, 16, 42, 40,
        0, 0, 0, 0, 0, 0, 0, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_knight_table[constants::BoardSize] = {
        -31, -24, -15, -3, -15, -9, -28, -27,
        -19, -16, -17, -19, -19, -6, -13, -19,
        -19, -9, -17, 7, 6, -16, -18, -23,
        -15, -2, 8, 13, 16, -1, -7, -11,
        -14, 4, 11, 20, 16, 15, -1, -13,
        -23, -17, 7, 11, 0, 9, -10, -19,
        -30, -20, -8, -5, -10, -14, -21, -25,
        -50, -29, -20, -20, -19, -28, -22, -46};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_bishop_table[constants::BoardSize] = {
        -13, -7, -11, 0, -3, 2, -15, -11,
        -8, -18, -11, -6, -1, -3, -13, -13,
        -13, 3, 1, 9, 7, -3, -8, -14,
        -7, -2, 5, 2, 3, 6, -9, -9,
        -1, 9, 2, 9, 7, -4, -4, -8,
        -3, 2, -2, 5, -1, 10, 2, -3,
        -17, -3, -5, -11, -2, -6, -5, -23,
        -14, -21, -17, -11, -14, -11, -11, -14};

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_rook_table[constants::BoardSize] = {
        11, 10, 5, 4, -4, 3, 4, -17,
        3, 1, 10, 7, 2, 2, 2, -4,
        1, 3, -1, 1, 1, -3, -6, -8,
        8, 6, 8, 8, 5, 1, -2, -7,
        10, 8, 9, 5, 3, 5, 1, 4,
        13, 13, 9, 4, 3, 6, 9, 3,
        17, 26, 17, 17, 8, 13, 14, 7,
        26, 17, 19, 13, 9, 9, 10, 13};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_queen_table[constants::BoardSize] = {
        -11, -15, -15, -39, -6, -13, -13, -19,
        -11, -4, -5, -4, -10, -4, -7, -8,
        -18, -18, 9, 2, 7, 9, 9, -9,
        -6, 0, 4, 23, 21, 9, 8, -2,
        -16, 1, -1, 22, 27, 22, 11, 6,
        -20, -7, 7, 6, 26, 16, 16, 2,
        -13, -3, -2, 9, 10, 30, 9, -2,
        -17, -7, -2, 7, 6, 4, -9, -5};

    // Table de correspondance pour le milieu de jeu
    static CONST_GUARD TUNABLE_TYPE *mg_tables[] = {
        mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table};

    static CONST_GUARD TUNABLE_TYPE *eg_tables[] = {
        eg_pawn_table, eg_knight_table, eg_bishop_table, eg_rook_table, eg_queen_table, eg_king_table};

    static const int pawnPhase = 0;
    static const int knightPhase = 1;
    static const int bishopPhase = 1;
    static const int rookPhase = 2;
    static const int queenPhase = 3;
    static const int totalPhase = 19;

    static constexpr int alphaBetaMargin = 90;

    static const int phase_values[] = {pawnPhase, knightPhase, bishopPhase, rookPhase, queenPhase, totalPhase};
    // ================ Mobility ================
    static CONSTEXPR_GUARD TUNABLE_TYPE knight_mob[] = {
        -28, 9, 21, 24, 31, 32, 34, 35,
        33};

    static CONSTEXPR_GUARD TUNABLE_TYPE bishop_mob[] = {
        -13, -4, 5, 9, 15, 20, 24, 27,
        31, 31, 37, 37, 38, 26};

    static CONSTEXPR_GUARD TUNABLE_TYPE rook_mob[] = {
        7, 18, 21, 26, 27, 35, 39, 45,
        49, 52, 57, 62, 69, 70, 68};

    static CONSTEXPR_GUARD TUNABLE_TYPE queen_mob[] = {
        39, 37, 38, 38, 39, 45, 47, 47,
        47, 54, 56, 59, 59, 64, 66, 68,
        71, 72, 79, 79, 81, 86, 87, 87,
        78, 88, 73, 72};

    // ================ Material ================
    static CONSTEXPR_GUARD std::array<TUNABLE_TYPE, constants::PieceTypeCount> pieces_score = {
        TUNABLE_TYPE(78), TUNABLE_TYPE(311), TUNABLE_TYPE(329),
        TUNABLE_TYPE(505), TUNABLE_TYPE(1026), TUNABLE_TYPE(10000)};

    // ================ Pawn structure ================
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesMgMalus = -1;
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesEgMalus = -22;

    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesMgMalus = -22;
    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesEgMalus = -4;

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_mg[] = {
        0, 6, 2, -7, 3, -8, 22, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_eg[] = {
        0, 2, 8, 28, 54, 121, 176, 0};

    // ================ King safety ================
    static CONSTEXPR_GUARD TUNABLE_TYPE openFileMalus = -43;
    static CONSTEXPR_GUARD TUNABLE_TYPE semiOpenFileMalus = -16;

    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesOpenFileMalus = -11;
    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesSemiOpenFileMalus = -3;

    // ================ Bishop pair ================
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairMgBonus = 51;
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairEgBonus = 38;

    // ================ Endgame / Mop-up ================
    static CONSTEXPR_GUARD TUNABLE_TYPE kingDistFromCenterBonus = 22;
    static CONSTEXPR_GUARD TUNABLE_TYPE closeKingBonus = -10;

    static CONSTEXPR_GUARD TUNABLE_TYPE maxDistBetweenKings = 11;

}