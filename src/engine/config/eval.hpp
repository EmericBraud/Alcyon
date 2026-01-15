#pragma once

#include <array>

#include "common/constants.hpp"
namespace engine::config::eval
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

    // ================ Mobility ================
    static constexpr int knight_mob[9] = {
        -20, -10, 0, 5, 10, 15, 20, 25, 30};

    static constexpr int bishop_mob[14] = {
        -20, -10, 0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65};

    static constexpr int rook_mob[15] = {
        -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55};

    static constexpr int queen_mob[28] = {
        -20, -15, -10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46};

    // ================ PST ================

    static constexpr int mg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 8, 10, 10, 8, 6, 6,
        8, 8, 12, 14, 14, 12, 8, 8,
        10, 10, 14, 16, 16, 14, 10, 10,
        12, 12, 16, 18, 18, 16, 12, 12,
        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr int mg_knight_table[constants::BoardSize] = {
        -30, -20, -10, -10, -10, -10, -20, -30,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -30, -20, -10, -10, -10, -10, -20, -30};
    static constexpr int mg_bishop_table[constants::BoardSize] = {
        -15, -10, -10, -10, -10, -10, -10, -15,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -15, -10, -10, -10, -10, -10, -10, -15};

    static constexpr int mg_rook_table[constants::BoardSize] = {
        0, 0, 0, 5, 5, 0, 0, 0,
        5, 10, 15, 15, 15, 15, 10, 5, // 2e rangée
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 15, 15, 15, 15, 10, 5, // 7e rangée (après miroir)
        0, 0, 0, 5, 5, 0, 0, 0};

    static constexpr int mg_queen_table[constants::BoardSize] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

    static constexpr int mg_king_table[constants::BoardSize] = {
        30, 40, 20, 0, 0, 20, 40, 30,
        20, 30, 10, 0, 0, 10, 30, 20,
        0, 10, -10, -20, -20, -10, 10, 0,
        -10, -20, -30, -40, -40, -30, -20, -10,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20,
        -20, -30, -40, -50, -50, -40, -30, -20};
    static constexpr int eg_king_table[constants::BoardSize] = {
        -10, -5, 0, -5, -5, 0, -5, -10,
        -5, -5, 0, 5, 7, 5, -5, -5,
        -5, -5, 5, 10, 10, 5, -5, -5,
        -5, 0, 10, 15, 15, 10, 0, -5,
        -5, 0, 10, 15, 15, 10, 0, -5,
        -10, -5, 5, 10, 10, 5, -5, -10,
        -15, -10, -5, 0, 0, -5, -10, -15,
        -20, -15, -10, -5, -5, -10, -15, -20};
    static constexpr int eg_pawn_table[constants::BoardSize] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 6, 7, 8, 8, 7, 6, 5,
        10, 12, 14, 16, 16, 14, 12, 10,
        18, 20, 22, 24, 24, 22, 20, 18,
        26, 28, 30, 32, 32, 30, 28, 26,
        34, 36, 38, 40, 40, 38, 36, 34,
        45, 45, 45, 45, 45, 45, 45, 45,
        0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr int eg_knight_table[constants::BoardSize] = {
        -30, -20, -10, -10, -10, -10, -20, -30,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 15, 20, 20, 15, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -20, -5, 0, 5, 5, 0, -5, -20,
        -30, -20, -10, -10, -10, -10, -20, -30};
    static constexpr int eg_bishop_table[constants::BoardSize] = {
        -15, -10, -10, -10, -10, -10, -10, -15,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 15, 15, 10, 5, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 0, 5, 5, 0, 0, -10,
        -15, -10, -10, -10, -10, -10, -10, -15};

    static constexpr int eg_rook_table[constants::BoardSize] = {
        0, 0, 0, 5, 5, 0, 0, 0,
        5, 10, 15, 15, 15, 15, 10, 5, // 2e rangée
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 15, 15, 15, 15, 10, 5, // 7e rangée (après miroir)
        0, 0, 0, 5, 5, 0, 0, 0};
    static constexpr int eg_queen_table[constants::BoardSize] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -5, 5, 10, 15, 15, 10, 5, -5,
        -5, 5, 10, 15, 15, 10, 5, -5,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

    static constexpr std::array<int, constants::PieceTypeCount> pieces_score = {
        100, 300, 300, 500, 900, 10000};

    // Table de correspondance pour le milieu de jeu
    static const int *mg_tables[] = {
        mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table};

    static const int *eg_tables[] = {
        eg_pawn_table, eg_knight_table, eg_bishop_table, eg_rook_table, eg_queen_table, eg_king_table};

    static const int pawnPhase = 0;
    static const int knightPhase = 1;
    static const int bishopPhase = 1;
    static const int rookPhase = 2;
    static const int queenPhase = 4;
    static const int totalPhase = 24;

    static const int phase_values[] = {pawnPhase, knightPhase, bishopPhase, rookPhase, queenPhase, totalPhase};

}