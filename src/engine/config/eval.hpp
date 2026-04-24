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
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 8, 10, 10, 8, 6, 6,
        8, 8, 12, 14, 14, 12, 8, 8,
        10, 10, 14, 16, 16, 14, 10, 10,
        12, 12, 16, 18, 18, 16, 12, 12,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_knight_table[constants::BoardSize] = {
        -30, -20, -10, -10, -10, -10, -20, -30,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -30, -20, -10, -10, -10, -10, -20, -30};
    static CONSTEXPR_GUARD TUNABLE_TYPE mg_bishop_table[constants::BoardSize] = {
        -15, -10, -10, -10, -10, -10, -10, -15,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -15, -10, -10, -10, -10, -10, -10, -15};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_rook_table[constants::BoardSize] = {
        0, 0, 0, 5, 5, 0, 0, 0,
        5, 10, 15, 15, 15, 15, 10, 5, // 2e rangée
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 15, 15, 15, 15, 10, 5, // 7e rangée (après miroir)
        0, 0, 0, 5, 5, 0, 0, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_queen_table[constants::BoardSize] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

    static CONSTEXPR_GUARD TUNABLE_TYPE mg_king_table[constants::BoardSize] = {
        30, 40, 20, 0, 0, 20, 40, 30,
        20, 30, 10, 0, 0, 10, 30, 20,
        0, 10, -10, -20, -20, -10, 10, 0,
        -10, -20, -30, -40, -40, -30, -20, -10,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_king_table[constants::BoardSize] = {
        -10, -5, 0, -5, -5, 0, -5, -10,
        -5, -5, 0, 5, 7, 5, -5, -5,
        -5, -5, 5, 10, 10, 5, -5, -5,
        -5, 0, 10, 15, 15, 10, 0, -5,
        -5, 0, 10, 15, 15, 10, 0, -5,
        -10, -5, 5, 10, 10, 5, -5, -10,
        -15, -10, -5, 0, 0, -5, -10, -15,
        -20, -15, -10, -5, -5, -10, -15, -20};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 6, 7, 8, 8, 7, 6, 5,
        10, 12, 14, 16, 16, 14, 12, 10,
        18, 20, 22, 24, 24, 22, 20, 18,
        26, 28, 30, 32, 32, 30, 28, 26,
        34, 36, 38, 40, 40, 38, 36, 34,
        45, 45, 45, 45, 45, 45, 45, 45,
        0, 0, 0, 0, 0, 0, 0, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_knight_table[constants::BoardSize] = {
        -30, -20, -10, -10, -10, -10, -20, -30,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -30, -20, -10, -10, -10, -10, -20, -30};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_bishop_table[constants::BoardSize] = {
        -15, -10, -10, -10, -10, -10, -10, -15,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -15, -10, -10, -10, -10, -10, -10, -15};

    static CONSTEXPR_GUARD TUNABLE_TYPE eg_rook_table[constants::BoardSize] = {
        0, 0, 0, 5, 5, 0, 0, 0,
        5, 10, 15, 15, 15, 15, 10, 5, // 2e rangée
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 15, 15, 15, 15, 10, 5, // 7e rangée (après miroir)
        0, 0, 0, 5, 5, 0, 0, 0};
    static CONSTEXPR_GUARD TUNABLE_TYPE eg_queen_table[constants::BoardSize] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -5, 5, 10, 15, 15, 10, 5, -5,
        -5, 5, 10, 15, 15, 10, 5, -5,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

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
    static CONSTEXPR_GUARD TUNABLE_TYPE knight_mob[9] = {
        -37, 8, 23, 26, 38, 43, 44, 43, 45};

    static CONSTEXPR_GUARD TUNABLE_TYPE bishop_mob[14] = {
        -11, 1, 12, 15, 23, 29, 28, 31, 36, 33, 37, 39, 38, 28};

    static CONSTEXPR_GUARD TUNABLE_TYPE rook_mob[15] = {
        21, 30, 30, 28, 30, 39, 41, 50, 58, 62, 68, 73, 81, 81, 86};

    static CONSTEXPR_GUARD TUNABLE_TYPE queen_mob[28] = {
        51, 53, 55, 56, 60, 67, 69, 66, 65, 71, 71, 72, 70, 73,
        73, 75, 79, 79, 89, 90, 92, 96, 99, 100, 89, 100, 76, 76};

    // ================ Material ================
    static CONSTEXPR_GUARD std::array<TUNABLE_TYPE, constants::PieceTypeCount> pieces_score = {
        TUNABLE_TYPE(91), TUNABLE_TYPE(376), TUNABLE_TYPE(394),
        TUNABLE_TYPE(607), TUNABLE_TYPE(1225), TUNABLE_TYPE(10000)};

    // ================ Pawn structure ================
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesMgMalus = 2;
    static CONSTEXPR_GUARD TUNABLE_TYPE doubledFilesEgMalus = -25;

    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesMgMalus = -26;
    static CONSTEXPR_GUARD TUNABLE_TYPE isolatedFilesEgMalus = -11;

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_mg[] = {
        0, -7, -8, -16, 5, -10, 31, 0};

    static CONSTEXPR_GUARD TUNABLE_TYPE passed_bonus_eg[] = {
        0, 29, 23, 40, 72, 160, 231, 0};

    // ================ King safety ================
    static CONSTEXPR_GUARD TUNABLE_TYPE openFileMalus = -46;
    static CONSTEXPR_GUARD TUNABLE_TYPE semiOpenFileMalus = -32;

    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesOpenFileMalus = -24;
    static CONSTEXPR_GUARD TUNABLE_TYPE heavyEnemiesSemiOpenFileMalus = -5;

    // ================ Bishop pair ================
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairMgBonus = 51;
    static CONSTEXPR_GUARD TUNABLE_TYPE bishopPairEgBonus = 57;

    // ================ Endgame / Mop-up ================
    static CONSTEXPR_GUARD TUNABLE_TYPE kingDistFromCenterBonus = 21;
    static CONSTEXPR_GUARD TUNABLE_TYPE closeKingBonus = -9;

    static CONSTEXPR_GUARD TUNABLE_TYPE maxDistBetweenKings = 14;

}