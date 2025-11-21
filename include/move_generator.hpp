#pragma once

#include "board.hpp"

namespace MoveGen
{
    extern std::array<U64, BOARD_SIZE> KnightAttacks;
    extern std::array<U64, BOARD_SIZE> KingAttacks;
    extern std::array<U64, BOARD_SIZE> RookMasks;
    extern std::array<U64, BOARD_SIZE> BishopMasks;
    extern std::array<U64, BOARD_SIZE> PawnAttacksWhite;
    extern std::array<U64, BOARD_SIZE> PawnAttacksBlack;
    extern std::array<U64, BOARD_SIZE> PawnPushWhite;
    extern std::array<U64, BOARD_SIZE> PawnPushBlack;
    extern std::array<U64, BOARD_SIZE> PawnPush2White;
    extern std::array<U64, BOARD_SIZE> PawnPush2Black;

    void initialize_bitboard_tables();
    void initialize_rook_masks();
    void initialize_bishop_masks();
    void initialize_pawn_masks();

    std::vector<Move> generate_pseudo_legal_moves(const Board &board, Color color);

    U64 generate_knight_moves(int from_sq, const Board &board);

    U64 generate_rook_moves(int from_sq, const Board &board);
}