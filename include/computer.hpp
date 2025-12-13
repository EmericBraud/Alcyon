#pragma once
#include "move_generator.hpp"

#define MAX_DEPTH 5
constexpr int INF = 1000000;
constexpr int MATE_SCORE = 100000;

class Computer
{
    Board &board;
    std::array<Move, MAX_DEPTH> move_stack;
    std::array<int, N_PIECES_TYPE_HALF> pieces_score = {
        100,   // Pawn
        300,   // Knight
        300,   // Bishop
        500,   // Tower
        900,   // Queen
        10000, // King
    };

    inline int eval() const
    {
        int score{0};
        for (int piece{PAWN}; piece <= KING; ++piece)
        {
            bitboard black_bitb = board.get_piece_bitboard(BLACK, piece), white_bitb = board.get_piece_bitboard(WHITE, piece);
            score += (std::popcount(white_bitb) - std::popcount(black_bitb)) * pieces_score[piece];
        }
        return score;
    }
    int eval_relative(Color side_to_move) const
    {
        int score = eval();
        return (side_to_move == WHITE) ? score : -score;
    }

public:
    int negamax(int depth, int alpha, int beta, int ply)
    {
        if (depth == 0)
        {
            return eval_relative(board.get_side_to_move());
        }

        const Color player = board.get_side_to_move();
        std::vector<Move> v = MoveGen::generate_pseudo_legal_moves(board, player);

        int legal_moves_count = 0;
        int best_score = -INF;

        for (Move &m : v)
        {
            board.play(m);

            if (MoveGen::is_king_attacked(board, player))
            {
                board.unplay(m);
                continue;
            }

            legal_moves_count++;

            int score = -negamax(depth - 1, -beta, -alpha, ply + 1);

            board.unplay(m);

            if (score >= beta)
            {
                return score;
            }

            if (score > best_score)
            {
                best_score = score;
                if (score > alpha)
                {
                    alpha = score;
                }
            }
        }

        if (legal_moves_count == 0)
        {
            if (MoveGen::is_king_attacked(board, player)) // Échec et Mat
            {
                return -MATE_SCORE + ply;
            }
            else
            {
                return 0;
            }
        }

        return best_score;
    }

    void play()
    {
        int depth = 5;
        int alpha = -INF;
        int beta = INF;

        int best_score = -INF;
        Move best_move;

        const Color player = board.get_side_to_move();
        std::vector<Move> v = MoveGen::generate_pseudo_legal_moves(board, player);

        for (Move &m : v)
        {
            board.play(m);
            if (MoveGen::is_king_attacked(board, player))
            {
                board.unplay(m);
                continue;
            }

            int score = -negamax(depth - 1, -beta, -alpha, 1);

            board.unplay(m);

            if (score > best_score)
            {
                best_score = score;
                best_move = m;
            }

            if (score > alpha)
            {
                alpha = score;
            }
        }
        board.play(best_move);
    }

public:
    Computer(Board &board) : board(board)
    {
    }

    int eval_position()
    {
        return negamax(MAX_DEPTH, -INF, INF, 0);
    }
};