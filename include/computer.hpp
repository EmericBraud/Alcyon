#pragma once
#include "move_generator.hpp"

#define MAX_DEPTH 8

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

    int alpha_beta_search(int depth, int alpha, int beta)
    {
        if (depth == MAX_DEPTH)
        {
            return eval();
        }
        const Color player{board.get_side_to_move()};
        std::vector<Move> v = MoveGen::generate_pseudo_legal_moves(board, player);
        int best_score;
        if (player == WHITE)
        {
            best_score = std::numeric_limits<int>::min();
            for (Move &m : v)
            {
                board.play(m);
                if (MoveGen::is_king_attacked(board))
                {
                    board.unplay(m);
                    continue;
                }
                best_score = std::max(best_score, alpha_beta_search(depth + 1, alpha, beta));
                board.unplay(m);
                if (best_score >= beta)
                {
                    return best_score;
                }
                alpha = std::max(alpha, best_score);
            }
        }
        else
        {
            best_score = std::numeric_limits<int>::max();
            for (Move &m : v)
            {
                board.play(m);
                if (MoveGen::is_king_attacked(board))
                {
                    board.unplay(m);
                    continue;
                }
                best_score = std::min(best_score, alpha_beta_search(depth + 1, alpha, beta));
                board.unplay(m);
                if (best_score <= alpha)
                {
                    return best_score;
                }
                beta = std::min(beta, best_score);
            }
        }
        return best_score;
    }

public:
    Computer(Board &board) : board(board)
    {
    }

    int eval_position()
    {
        return alpha_beta_search(0, std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
    }
};