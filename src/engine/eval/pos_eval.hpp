#pragma once

#include <array>

#include "common/mask.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"

#include "engine/config/eval.hpp"
#include "engine/eval/virtual_board.hpp"

namespace Eval
{

    struct PawnMasks
    {
        std::array<U64, 8> file;
        std::array<U64, 8> adjacent;
        std::array<std::array<U64, 64>, 2> passed;
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

    int evaluate_castling_and_safety(Color color, const VBoard &board);

    void evaluate_pawns(Color color, const VBoard &board, int &mg, int &eg);
    int eval(const VBoard &board, int alpha, int beta);

    template <Color Us>
    inline int eval_relative(const VBoard &board, int alpha, int beta)
    {
        int score = eval(board, alpha, beta);
        return (Us == WHITE) ? score : -score;
    }

    inline int get_piece_score(int piece)
    {
        return engine::config::eval::pieces_score[piece];
    }

    void print_pawn_stats();

    inline int king_distance(int sq1, int sq2)
    {
        int dx = std::abs((sq1 & 7) - (sq2 & 7));
        int dy = std::abs((sq1 >> 3) - (sq2 >> 3));
        return std::max(dx, dy);
    }

    template <Color Us>
    int lazy_eval_relative(const VBoard &board)
    {
        const EvalState &state = board.get_eval_state();
        const int mg_score = (state.mg_pst[WHITE] + state.pieces_val[WHITE]) -
                             (state.mg_pst[BLACK] + state.pieces_val[BLACK]);
        const int eg_score = (state.eg_pst[WHITE] + state.pieces_val[WHITE]) -
                             (state.eg_pst[BLACK] + state.pieces_val[BLACK]);
        const int base_score = (mg_score * state.phase + eg_score * (engine::config::eval::totalPhase - state.phase)) / engine::config::eval::totalPhase;
        return Us == WHITE ? base_score : -base_score;
    }
}