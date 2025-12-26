#pragma once

#include "move_generator.hpp"

namespace eval
{
    static const int pawnPhase = 0;
    static const int knightPhase = 1;
    static const int bishopPhase = 1;
    static const int rookPhase = 2;
    static const int queenPhase = 4;
    static const int totalPhase = 24;

    static constexpr int mg_pawn_table[64] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr int mg_knight_table[64] = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50};

    static constexpr int mg_bishop_table[64] = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20};

    static constexpr int mg_rook_table[64] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0};

    static constexpr int mg_queen_table[64] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

    static constexpr int mg_king_table[64] = {
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30};

    static constexpr int eg_king_table[64] = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50};

    static constexpr std::array<int, N_PIECES_TYPE_HALF> pieces_score = {
        100, 300, 300, 500, 900, 10000};

    // Table de correspondance pour le milieu de jeu
    static const int *mg_tables[] = {
        mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table};

    inline int evaluate_king_safety(Color color, int king_sq, const Board &board)
    {
        int penalty = 0;
        bitboard pawns = board.get_piece_bitboard(color, PAWN);
        bitboard shield_area = (color == WHITE) ? (1ULL << (king_sq + 7) | 1ULL << (king_sq + 8) | 1ULL << (king_sq + 9))
                                                : (1ULL << (king_sq - 7) | 1ULL << (king_sq - 8) | 1ULL << (king_sq - 9));

        if (std::popcount(pawns & shield_area) == 0)
            penalty -= 50;

        return penalty;
    }

    inline int evaluate_pawn_structure(Color color, const Board &board)
    {
        int score = 0;
        bitboard pawns = board.get_piece_bitboard(color, PAWN);
        for (int i = 0; i < 8; ++i)
        {
            bitboard file_mask = 0x0101010101010101ULL << i;
            if (std::popcount(pawns & file_mask) > 1)
                score -= 20;
        }
        return score;
    }

    inline int eval(const Board &board)
    {
        int mg[2] = {0, 0};
        int eg[2] = {0, 0};
        int gamePhase = 0;

        for (int color = WHITE; color <= BLACK; ++color)
        {
            for (int piece = PAWN; piece <= KING; ++piece)
            {
                bitboard bb = board.get_piece_bitboard((Color)color, piece);
                const int *mg_table = mg_tables[piece]; // RÉCUPÉRATION DE LA TABLE CORRECTE

                while (bb)
                {
                    int sq = std::countr_zero(bb);
                    int mirror_sq = (color == WHITE) ? sq : sq ^ 56;

                    // Ajout du score matériel + PST spécifique à la pièce
                    mg[color] += pieces_score[piece] + mg_table[mirror_sq];

                    // En finale, seul le Roi change radicalement de table
                    if (piece == KING)
                    {
                        eg[color] += pieces_score[piece] + eg_king_table[mirror_sq];
                    }
                    else
                    {
                        eg[color] += pieces_score[piece] + mg_table[mirror_sq];
                    }

                    if (piece != PAWN && piece != KING)
                    {
                        gamePhase += (piece == KNIGHT) ? knightPhase : (piece == BISHOP) ? bishopPhase
                                                                   : (piece == ROOK)     ? rookPhase
                                                                                         : queenPhase;
                    }

                    if (piece == KING)
                        mg[color] += evaluate_king_safety((Color)color, sq, board);

                    bb &= (bb - 1);
                }
            }
            mg[color] += evaluate_pawn_structure((Color)color, board);
        }

        int mgScore = mg[WHITE] - mg[BLACK];
        int egScore = eg[WHITE] - eg[BLACK];
        int phase = std::min(gamePhase, totalPhase);

        return (mgScore * phase + egScore * (totalPhase - phase)) / totalPhase;
    }

    inline int eval_relative(Color side_to_move, const Board &board)
    {
        int score = eval(board);
        return (side_to_move == WHITE) ? score : -score;
    }

    inline int get_piece_score(int piece)
    {
        return pieces_score[piece];
    }
}