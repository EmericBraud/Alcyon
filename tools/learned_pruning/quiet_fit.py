"""Coups calmes tardifs : combien de noeuds un modele retire-t-il sans perdre les coups utiles ?

    python3 quiet_fit.py <dossier de dumps game_NNN.bin> [every]

Dumps : ALCYON_QUIET_DUMP (search::QuietRecord), un fichier par partie
(context_bias.py play, TT conservee). Parties 0-259 en entrainement, le
reste en test. Poids de tirage 1 / min(1, 2^(depth-1) / every). Cout d'un
coup = son sous-arbre (les sous-arbres imbriques se recouvrent : les parts
sont relatives au cout total des coups calmes tardifs tires, pas a l'arbre).

Trois classements du moins au plus utile : hasard, "rang" (profondeur x
rang du coup, ce que voient la LMR et la LMP), et un modele a gradient
boosting sur toutes les features. Pour chacun : part du cout retirable en
ne perdant que 0.1 / 0.5 / 1 / 2 % des coups utiles (en poids).
"""
import glob
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

FIELDS = ("depth ply rank is_pv cut_node improving gives_check eval_alpha order_score history "
          "piece to_rank tt_move halfmove pieces alpha useful score_alpha").split()
DTYPE = np.dtype([("subtree", "<i8")] + [(n, "<i4") for n in FIELDS])
assert DTYPE.itemsize == 80
EVAL_NONE = 1 << 30
LOSSES = [0.001, 0.005, 0.01, 0.02]


def load(folder, every):
    parts = {}
    for path in sorted(glob.glob(os.path.join(folder, "game_*.bin"))):
        g = int(os.path.basename(path)[5:8])
        parts[g] = np.fromfile(path, dtype=DTYPE)
    tr = np.concatenate([r for g, r in parts.items() if g < 260])
    te = np.concatenate([r for g, r in parts.items() if g >= 260])
    return tr, te


def weight(r, every):
    return 1.0 / np.minimum(1.0, 2.0 ** (np.minimum(r["depth"], 60) - 1) / every)


def features(r):
    ea = r["eval_alpha"].astype(np.float64)
    known = ea != EVAL_NONE
    ea = np.where(known, np.clip(ea, -1500, 1500), 0)
    cols = [r["depth"], r["ply"], np.log1p(r["rank"]), r["is_pv"], r["cut_node"], r["improving"],
            r["gives_check"], ea, known, np.sign(r["order_score"]) * np.log1p(np.abs(r["order_score"])),
            np.sign(r["history"]) * np.log1p(np.abs(r["history"])), r["piece"], r["to_rank"], r["tt_move"],
            r["halfmove"], r["pieces"]]
    return np.column_stack(cols).astype(np.float32)


def curve(score, useful, cost, w):
    """Retire les coups par score croissant (le moins utile d'abord) : part du
    cout retire quand la part des coups utiles perdus atteint chaque seuil."""
    order = np.argsort(score, kind="stable")
    u = np.cumsum((useful * w)[order]) / (useful * w).sum()
    c = np.cumsum((cost * w)[order]) / (cost * w).sum()
    return [c[np.searchsorted(u, x, side="right") - 1] if (u <= x).any() else 0.0 for x in LOSSES]


def main():
    folder = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    tr, te = load(folder, every)
    wtr, wte = weight(tr, every), weight(te, every)
    ytr, yte = tr["useful"], te["useful"]
    print(f"train {len(tr)} coups, test {len(te)} ; utiles (pondere) {np.average(yte, weights=wte):.2%}")
    cost = te["subtree"].astype(np.float64)

    # Rang : taux d'utilite empirique par (profondeur, rang) appris en train.
    key = lambda r: np.minimum(r["depth"], 20) * 256 + np.minimum(r["rank"], 255)  # noqa: E731
    ktr, kte = key(tr), key(te)
    num = np.bincount(ktr, weights=wtr * ytr, minlength=21 * 256)
    den = np.bincount(ktr, weights=wtr, minlength=21 * 256)
    rank_rate = (num + 1e-3) / (den + 1)

    xtr, xte = features(tr), features(te)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(tr), min(len(tr), 4_000_000), replace=False)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63)
    model.fit(xtr[sub], ytr[sub], sample_weight=wtr[sub])
    p = model.predict_proba(xte)[:, 1]

    head = "  ".join(f"perte {x:.1%}" for x in LOSSES)
    print(f"\npart du cout des coups calmes tardifs retirable, selon la part des coups utiles perdus :\n  {'classement':22s}{head}")
    for name, s in [("hasard", rng.random(len(te))), ("rang (depth x rang)", rank_rate[kte]), ("modele (toutes features)", p)]:
        print(f"  {name:22s}" + "  ".join(f"{v:10.1%}" for v in curve(s, yte, cost, wte)))
    for lo, hi in [(1, 3), (4, 6), (7, 64)]:
        m = (te["depth"] >= lo) & (te["depth"] <= hi)
        print(f"\n  depth {lo}-{hi} ({m.sum()} coups, utiles {np.average(yte[m], weights=wte[m]):.2%}, "
              f"{(cost[m] * wte[m]).sum() / (cost * wte).sum():.0%} du cout)")
        for name, s in [("rang", rank_rate[kte]), ("modele", p)]:
            print(f"    {name:20s}" + "  ".join(f"{v:10.1%}" for v in curve(s[m], yte[m], cost[m], wte[m])))


if __name__ == "__main__":
    main()
