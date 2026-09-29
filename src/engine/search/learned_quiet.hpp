#pragma once

// Coups calmes tardifs (etape QUIETS du MovePicker) : P(le coup monte alpha),
// par une table (profondeur x rang) ou par le MLP l0 de
// tools/learned_pruning/quiet_mlp.py (12 experts profondeur x cut/all,
// recalibre). Doit reproduire quiet_fit.features et quiet_mlp.Net
// (quiet_export.py check). Decision dans negamax.cpp (learned_quiet_mode).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include "learned_quiet_weights.hpp"

namespace learned_quiet
{
    struct Input
    {
        int depth, ply, rank, is_pv, cut_node, improving, gives_check;
        int eval_alpha; // eval du noeud - alpha
        bool eval_known;
        int order_score, history, piece, to_rank, tt_move, halfmove, pieces;
    };

    inline int expert(int depth, bool cut)
    {
        const int d = depth <= 3 ? depth - 1 : depth <= 6 ? 3 : depth <= 9 ? 4 : 5;
        return 2 * d + (cut ? 1 : 0);
    }

    inline float signlog(float v) { return v < 0 ? -std::log1p(-v) : std::log1p(v); }

    inline float log_cost(int depth, int rank) { return kLogCost[std::min(depth, kMaxD)][std::min(rank, kMaxR)]; }
    inline float log_rate(int depth, int rank) { return kLogRate[std::min(depth, kMaxD)][std::min(rank, kMaxR)]; }

    // Logit recalibre de P(utile).
    inline float logit(const Input &in, const std::array<std::uint8_t, 1024> &l0)
    {
        const float raw[kX] = {
            float(in.depth), float(in.ply), std::log1p(float(in.rank)), float(in.is_pv), float(in.cut_node),
            float(in.improving), float(in.gives_check),
            in.eval_known ? std::clamp(float(in.eval_alpha), -1500.0f, 1500.0f) : 0.0f, in.eval_known ? 1.0f : 0.0f,
            signlog(float(in.order_score)), signlog(float(in.history)), float(in.piece), float(in.to_rank),
            float(in.tt_move), float(in.halfmove), float(in.pieces)};
        float x[kX];
        for (int i = 0; i < kX; ++i)
            x[i] = (raw[i] - kMean[i]) / kStd[i];

        float h[kHidden];
        for (int j = 0; j < kHidden; ++j)
            h[j] = kL0B[j];
        for (int i = 0; i < 1024; ++i)
            if (l0[i])
                for (int j = 0; j < kHidden; ++j)
                    h[j] += kL0WT[i][j] * l0[i];
        for (int j = 0; j < kHidden; ++j)
            h[j] = std::max(h[j], 0.0f);

        const int e = expert(in.depth, in.cut_node);
        float z = kLinB[e] + kH2B[e] + kPrior[e];
        for (int i = 0; i < kX; ++i)
            z += kLinW[e][i] * x[i];
        for (int k = 0; k < 32; ++k)
        {
            float a = kH1B[e][k];
            for (int j = 0; j < kHidden; ++j)
                a += kH1W[e][k][j] * h[j];
            for (int i = 0; i < kX; ++i)
                a += kH1W[e][k][kHidden + i] * x[i];
            z += kH2W[e][k] * std::max(a, 0.0f);
        }
        return z;
    }
}
