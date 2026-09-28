#pragma once

// Elagage appris (docs/learned-pruning.md). Deux modeles de P(fail-high) =
// sigmoid(z) sur les memes 75 entrees -- le vecteur de fit.design(m, x, "b"),
// que design() doit reconstruire EXACTEMENT (meme ordre, memes bornes,
// memes echelles) :
//  - logit()     : regression par profondeur (tools/learned_pruning/export.py)
//  - mlp_logit() : train_mlp.Net, qui ajoute l'entree l0 de la pile NNUE
//                  (tools/learned_pruning/export_mlp.py)
// Le dump enregistre le z du modele actif (PruneRecord::learned_z) ;
// `export.py check` et `export_mlp.py check` verifient la parite.

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>

#include "engine/search/learned_prune_mlp_weights.hpp"
#include "engine/search/learned_prune_weights.hpp"
#include "engine/search/worker.hpp"

namespace learned_prune
{
    constexpr int kScalars = 37;
    constexpr int kDesign = 1 + 2 * kScalars;
    static_assert(kInputs == kDesign, "reexporter les poids (fit.features a change)");
    static_assert(learned_prune_mlp::kDesign == kDesign, "reexporter le MLP (fit.features a change)");
    constexpr int kPieceCp[5] = {100, 300, 300, 500, 900};

    inline float clip_pawns(int v, int lim = 1500) { return std::clamp(v, -lim, lim) / 100.0f; }

    // [m, x..., m*x...] avec m = eval - beta en pions.
    inline void design(const search::PruneRecord &r, float d[kDesign])
    {
        const bool found = r.tt_found != 0;
        const bool prev2_ok = r.eval_prev2 != SearchWorker::kEvalNone;
        float *x = d + 1;
        int i = 0;
        x[i++] = prev2_ok;
        x[i++] = prev2_ok ? clip_pawns(r.static_eval - r.eval_prev2) : 0.0f;
        x[i++] = found;
        x[i++] = found ? clip_pawns(r.tt_score - r.static_eval) : 0.0f;
        x[i++] = found ? static_cast<float>(std::clamp(r.tt_depth - r.depth, -10, 10)) : 0.0f;
        for (int f = 0; f < 3; ++f)
            x[i++] = found && r.tt_flag == f;
        x[i++] = r.cut_node;
        x[i++] = r.allow_null;
        x[i++] = std::min(r.halfmove, 100) / 100.0f;
        x[i++] = r.stm;
        x[i++] = r.ply / 10.0f;
        x[i++] = r.depth;
        x[i++] = r.prev_to_piece != NO_PIECE;
        for (int p = 0; p < 6; ++p)
            x[i++] = r.prev_from_piece == p;
        for (int p = 0; p < 5; ++p)
            x[i++] = r.prev_to_piece == p;
        int material = 0;
        for (int p = 0; p < 5; ++p)
        {
            x[i++] = r.us[p];
            material += r.us[p] * kPieceCp[p];
        }
        for (int p = 0; p < 5; ++p)
        {
            x[i++] = r.them[p];
            material -= r.them[p] * kPieceCp[p];
        }
        x[i++] = material / 100.0f;

        const float m = clip_pawns(r.static_eval - r.beta);
        d[0] = m;
        for (int k = 0; k < kScalars; ++k)
            d[1 + kScalars + k] = m * x[k];
    }

    inline float logit(const search::PruneRecord &r)
    {
        float d[kDesign];
        design(r, d);
        const float *w = kWeights[std::min(r.depth, kSlices) - 1];
        float z = w[0];
        for (int k = 0; k < kDesign; ++k)
            z += w[1 + k] * d[k];
        return z;
    }

    // train_mlp.Net.forward, normalisation et 1/127 replies dans les poids.
    // l0 est a ~90 % nul : on ne parcourt que ses octets non nuls (blocs de 8
    // nuls sautes d'un test) et on ajoute leur ligne de kL0WT -- ~100 x 16
    // MAC au lieu de 1024 x 16, meme resultat. Le produit dense coutait 33 %
    // de nps. ponytail: float ; int8 SIMD si le nps en souffre encore.
    inline float mlp_logit(const search::PruneRecord &r, const std::array<std::uint8_t, 1024> &l0)
    {
        using namespace learned_prune_mlp;
        float d[kDesign];
        design(r, d);

        float z = kLinB;
        for (int k = 0; k < kDesign; ++k)
            z += kLinW[k] * d[k];

        float h[kHidden];
        for (int j = 0; j < kHidden; ++j)
            h[j] = kL0B[j];
        for (int b = 0; b < 1024; b += 8)
        {
            std::uint64_t block;
            std::memcpy(&block, l0.data() + b, 8);
            while (block)
            {
                const int i = b + std::countr_zero(block) / 8;
                const float v = l0[i];
                for (int j = 0; j < kHidden; ++j)
                    h[j] += v * kL0WT[i][j];
                block &= ~(std::uint64_t{0xFF} << (std::countr_zero(block) / 8 * 8));
            }
        }
        for (int j = 0; j < kHidden; ++j)
            h[j] = std::max(h[j], 0.0f);

        const int depth_idx = std::min(r.depth, kDepths) - 1;
        float out = kH2B;
        for (int k = 0; k < kHead; ++k)
        {
            float a = kH1B[k] + kH1W[k][kHidden + kDesign + depth_idx];
            for (int j = 0; j < kHidden; ++j)
                a += kH1W[k][j] * h[j];
            for (int i = 0; i < kDesign; ++i)
                a += kH1W[k][kHidden + i] * d[i];
            out += kH2W[k] * std::max(a, 0.0f);
        }
        return z + out;
    }
}
