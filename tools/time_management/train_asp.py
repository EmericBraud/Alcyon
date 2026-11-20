"""Fenetre d'aspiration apprise : predit l'ampleur de la variation du score a
l'iteration suivante, pour centrer une fenetre plus etroite quand la position
est calme et plus large quand elle est agitee.

Entree : TSV de gen_tm_data.py (un score par iteration). Cible :
log(1 + |score(d+1) - score(d)|). Le moteur utilise delta = c * exp(pred),
c regle en parties. Ecrit un rapport et les poids (meme format que tm_mlp).

Usage : python train_asp.py DATA.tsv OUT_DIR
"""

import json
import os
import sys

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

FEATURES = ["depth", "stable", "frac_best", "frac_second", "abs_score", "abs_drop",
            "root_moves", "sel_minus_depth", "pieces"]


def main():
    data, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    rows = [l.rstrip("\n").split("\t") for l in open(data)]
    a = np.array([r[:14] for r in rows if len(r) == 16], dtype=np.float64)
    g, ply, d, el, soft, nodes, st, fb, fs, sc, pv, rm, sel, pc = a.T
    k = g * 1000 + ply
    same = np.r_[k[1:] == k[:-1], False]
    nxt = np.r_[sc[1:], 0]
    m = same & (d >= 5) & (np.abs(sc) < 1500) & (np.abs(nxt) < 1500)
    scc = np.clip(sc, -2000, 2000)
    pvc = np.clip(pv, -2000, 2000)
    X = np.column_stack([d, np.minimum(st, 20), fb, fs, np.abs(scc), np.abs(pvc - scc), rm, sel - d, pc])[m]
    dl = np.abs(nxt - sc)[m]
    y = np.log1p(dl)
    test = g[m] % 5 == 0
    sc_ = StandardScaler().fit(X[~test])
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=200, early_stopping=True,
                       random_state=0).fit(sc_.transform(X[~test]), y[~test])
    pred = mlp.predict(sc_.transform(X[test]))
    rep = {"rows": int(m.sum()), "features": FEATURES,
           "mse": float(np.mean((pred - y[test]) ** 2)), "mse_const": float(np.var(y[test]))}
    cur = np.where(d[m][test] >= 12, 20, np.where(d[m][test] >= 8, 41, 16))
    rep["current"] = {"coverage": float((dl[test] <= cur).mean()), "mean_width": float(cur.mean())}
    for c in [2, 3, 4, 6, 8]:
        w = np.maximum(8, c * np.exp(pred))
        rep[f"c{c}"] = {"coverage": float((dl[test] <= w).mean()), "mean_width": float(w.mean())}
    np.savez(os.path.join(out_dir, "asp_mlp.npz"), mean=sc_.mean_, scale=sc_.scale_,
             **{f"W{i}": w for i, w in enumerate(mlp.coefs_)},
             **{f"b{i}": b for i, b in enumerate(mlp.intercepts_)})
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
