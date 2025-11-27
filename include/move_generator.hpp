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

    extern std::array<Magic, BOARD_SIZE> RookMagics;
    extern std::array<Magic, BOARD_SIZE> BishopMagics;

    extern std::array<U64, ROOK_ATTACKS_SIZE> RookAttacks;
    extern std::array<U64, BISHOP_ATTACKS_SIZE> BishopAttacks;

    /// @brief Used for processing rook attacks tables (unused on prod)
    extern std::vector<U64>
        RookAttacksProcessing;
    /// @brief Used for processing bishop attacks tables (unused on prod)
    extern std::vector<U64> BishopAttacksProcessing;

    void initialize_bitboard_tables();
    inline U64 get_moves_mask(const Board &board, const int sq, const Color color, const Piece piece_type);
    U64 get_moves_mask(const Board &board, const int sq);
    void initialize_rook_masks();
    void initialize_bishop_masks();
    void initialize_pawn_masks();

    std::vector<Move> generate_pseudo_legal_moves(const Board &board, Color color);

    U64 generate_knight_moves(int from_sq, const Board &board);
    U64 generate_rook_moves(int from_sq, const Board &board);

    U64 generate_bishop_moves(int from_sq, const Board &board);

    void export_attack_table(const std::array<MoveGen::Magic, BOARD_SIZE> m_array, bool is_rook);
    void run_magic_searcher();

    /// @brief Gets the piece attack's sizes to then write them as static in the code (unused on prod)
    /// @param is_rook rook / bishop
    void get_sizes(bool is_rook);

    void load_magics(bool is_rook);

    void load_magics();
    void load_attacks_rook();
    void load_attacks_bishop();
    void load_attacks();
}

U64 generate_sliding_attack(int sq, U64 occupancy, bool is_rook);
