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

    struct Magic
    {
        U64 mask;
        U64 magic;
        int shift;
        long unsigned int index_start;
    };

    // 3. Déclarations des Données Magiques
    extern std::array<Magic, 64> RookMagics;
    extern std::array<Magic, 64> BishopMagics;

    void initialize_bitboard_tables();
    void initialize_rook_masks();
    void initialize_bishop_masks();
    void initialize_pawn_masks();

    std::vector<Move> generate_pseudo_legal_moves(const Board &board, Color color);

    U64 generate_knight_moves(int from_sq, const Board &board);

    U64 generate_rook_moves(int from_sq, const Board &board);

    void export_attack_table(std::array<MoveGen::Magic, BOARD_SIZE> m_array, bool is_rook);

    void run_magic_searcher();
}