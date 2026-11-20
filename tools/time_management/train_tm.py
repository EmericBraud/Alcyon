"""Entraine le predicteur de la gestion du temps : P(le coup change si on continue).

Entree : TSV de gen_tm_data.py. Une ligne par iteration terminee.
Cible : le coup de l'iteration differe du coup final de la recherche longue.
Separation train/test par partie. Compare la table de stabilite actuelle,
une regression logistique et un MLP ; ecrit un rapport JSON et les poids.

Usage : python train_tm.py DATA.tsv OUT_DIR
"""

import json
import math
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

COLS = ["game", "ply", "depth", "elapsed", "soft", "nodes", "stable", "frac_best", "frac_second",
        "score", "prev_score", "root_moves", "seldepth", "pieces", "move", "final"]
FEATURES = ["depth", "stable", "frac_best", "frac_second", "abs_score", "drop", "abs_drop",
            "root_moves", "sel_minus_depth", "pieces", "log_nodes"]


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != len(COLS):
                continue
            rows.append(p)
    a = np.array(rows, dtype=object)
    d = {c: a[:, i] for i, c in enumerate(COLS)}
    num = {c: d[c].astype(np.float64) for c in COLS if c not in ("move", "final")}
    y = (d["move"] != d["final"]).astype(np.int64)
    score = np.clip(num["score"], -2000, 2000)
    prev = np.clip(num["prev_score"], -2000, 2000)
    X = np.column_stack([
        num["depth"], np.minimum(num["stable"], 20), num["frac_best"], num["frac_second"],
        np.abs(score), prev - score, np.abs(prev - score), num["root_moves"],
        num["seldepth"] - num["depth"], num["pieces"], np.log1p(num["nodes"]),
    ])
    # La derniere iteration de chaque recherche est le coup final par definition.
    key = num["game"] * 1000 + num["ply"]
    last = np.r_[key[1:] != key[:-1], True]
    keep = ~last & (num["depth"] >= 4)
    return X[keep], y[keep], num["game"][keep].astype(np.int64), np.minimum(num["stable"], 4)[keep].astype(int)


def main():
    data, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    X, y, games, stable4 = load(data)
    test = games % 5 == 0
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    rep = {"rows": int(len(y)), "train": int((~test).sum()), "test": int(test.sum()),
           "base_rate": float(y.mean()), "features": FEATURES}

    # Reference : la table actuelle (taux par duree de stabilite, appris sur train).
    rates = np.array([ytr[stable4[~test] == k].mean() if (stable4[~test] == k).any() else ytr.mean()
                      for k in range(5)])
    p_tab = np.clip(rates[stable4[test]], 1e-4, 1 - 1e-4)
    rep["stability_table"] = {"rates": rates.tolist(), "auc": float(roc_auc_score(yte, p_tab)),
                              "logloss": float(log_loss(yte, p_tab))}

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    rep["logistic"] = {"auc": float(roc_auc_score(yte, p_lr)), "logloss": float(log_loss(yte, p_lr)),
                       "coef": dict(zip(FEATURES, lr.coef_[0].tolist())), "intercept": float(lr.intercept_[0])}

    best = None
    for hidden in [(16,), (32, 16), (64, 32)]:
        mlp = MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-3, max_iter=300, early_stopping=True,
                            random_state=0).fit(sc.transform(Xtr), ytr)
        p = mlp.predict_proba(sc.transform(Xte))[:, 1]
        r = {"auc": float(roc_auc_score(yte, p)), "logloss": float(log_loss(yte, p))}
        rep[f"mlp_{'x'.join(map(str, hidden))}"] = r
        if best is None or r["logloss"] < best[1]["logloss"]:
            best = (mlp, r, hidden)

    mlp, _, hidden = best
    np.savez(os.path.join(out_dir, "tm_mlp.npz"), mean=sc.mean_, scale=sc.scale_,
             **{f"W{i}": w for i, w in enumerate(mlp.coefs_)},
             **{f"b{i}": b for i, b in enumerate(mlp.intercepts_)})
    rep["best_mlp"] = list(hidden)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
