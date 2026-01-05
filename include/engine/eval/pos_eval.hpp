#pragma once

#include "core/move/move_generator.hpp"
#include "engine/eval/pawn_entry.hpp"

namespace Eval
{

    static constexpr int knight_mob[9] = {
        -20, -10, 0, 5, 10, 15, 20, 25, 30};

    static constexpr int bishop_mob[14] = {
        -20, -10, 0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65};

    static constexpr int rook_mob[15] = {
        -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55};

    static constexpr int queen_mob[28] = {
        -20, -15, -10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46};

    struct PawnMasks
    {
        std::array<bitboard, 8> file;
        std::array<bitboard, 8> adjacent;
        std::array<std::array<bitboard, 64>, 2> passed;
    };
    constexpr PawnMasks generate_masks()
    {
        PawnMasks m = {};

        // 1. Colonnes et colonnes adjacentes
        for (int f = 0; f < 8; f++)
        {
            m.file[f] = 0x0101010101010101ULL << f;
        }
        for (int f = 0; f < 8; f++)
        {
            if (f > 0)
                m.adjacent[f] |= m.file[f - 1];
            if (f < 7)
                m.adjacent[f] |= m.file[f + 1];
        }

        // 2. Pions passés
        for (int sq = 0; sq < 64; sq++)
        {
            int f = sq % 8;
            int r = sq / 8;

            for (int r_target = 0; r_target < 8; r_target++)
            {
                for (int f_target = 0; f_target < 8; f_target++)
                {
                    // Si la colonne cible est f-1, f, ou f+1
                    if (f_target >= f - 1 && f_target <= f + 1)
                    {
                        int target_sq = r_target * 8 + f_target;
                        if (r_target > r)
                            m.passed[0][sq] |= (1ULL << target_sq); // WHITE
                        if (r_target < r)
                            m.passed[1][sq] |= (1ULL << target_sq); // BLACK
                    }
                }
            }
        }
        return m;
    }

    static constexpr PawnMasks masks = generate_masks();

    int evaluate_castling_and_safety(Color color, const Board &board);

    void evaluate_pawns(Color color, const Board &board, int &mg, int &eg);
    int eval(const Board &board, int alpha, int beta);

    inline int eval_relative(Color side_to_move, const Board &board, int alpha, int beta)
    {
        int score = eval(board, alpha, beta);
        return (side_to_move == WHITE) ? score : -score;
    }

    inline int get_piece_score(int piece)
    {
        return pieces_score[piece];
    }

    void print_pawn_stats();

    int lazy_eval_relative(const Board &board, Color us);
}