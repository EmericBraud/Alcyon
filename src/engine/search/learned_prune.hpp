#pragma once

// Elagage appris (docs/learned-pruning.md) : regression logistique par
// profondeur, P(fail-high) = sigmoid(z). Les poids viennent de
// tools/learned_pruning/export.py ; logit() doit reconstruire EXACTEMENT
// le vecteur de fit.design(m, x, "b") -- meme ordre, memes bornes, memes
// echelles. Le dump enregistre z (PruneRecord::learned_z) pour que fit.py
// verifie la parite sur des noeuds reels.

#include <algorithm>

#include "engine/search/learned_prune_weights.hpp"
#include "engine/search/worker.hpp"

namespace learned_prune
{
    constexpr int kScalars = 37;
    static_assert(kInputs == 1 + 2 * kScalars, "reexporter les poids (fit.features a change)");
    constexpr int kPieceCp[5] = {100, 300, 300, 500, 900};

    inline float clip_pawns(int v, int lim = 1500) { return std::clamp(v, -lim, lim) / 100.0f; }

    inline float logit(const search::PruneRecord &r)
    {
        const bool found = r.tt_found != 0;
        const bool prev2_ok = r.eval_prev2 != SearchWorker::kEvalNone;
        float x[kScalars];
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
        const float *w = kWeights[std::min(r.depth, kSlices) - 1];
        float z = w[0] + w[1] * m;
        for (int k = 0; k < kScalars; ++k)
            z += (w[2 + k] + w[2 + kScalars + k] * m) * x[k];
        return z;
    }
}
