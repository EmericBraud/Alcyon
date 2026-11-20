"""Verifie que les entrees normalisees ont la meme distribution a toutes les cadences.

Le budget souple sert de proxy de la cadence (5+0.05 -> ~0.1 s, 60+0.6 -> ~2 s).
Pour chaque tranche de budget : quantiles 10/50/90 de chaque entree, et taux de
la cible. Les entrees absolues (profondeur, log noeuds) sont affichees pour
comparaison : elles doivent, elles, deriver avec la cadence.

Usage : python check_invariance.py DATA.tsv [DATA.tsv...]
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("TM_FEATURES", "norm")
import train_tm  # noqa: E402


def main():
    rows = []
    for path in sys.argv[1:]:
        with open(path) as f:
            rows += [l.rstrip("\n").split("\t") for l in f]
    rows = [r for r in rows if len(r) == len(train_tm.COLS)]
    a = np.array(rows, dtype=object)
    num = {c: a[:, i].astype(np.float64) for i, c in enumerate(train_tm.COLS) if c not in ("move", "final")}

    tmp = "/tmp/_inv.tsv"
    with open(tmp, "w") as f:
        f.write("\n".join("\t".join(r) for r in rows) + "\n")
    X, y, _, _ = train_tm.load(tmp)

    # Memes filtres que load() pour aligner soft / profondeur / noeuds.
    key = num["game"] * 1000 + num["ply"]
    last = np.r_[key[1:] != key[:-1], True]
    keep = ~last & (num["depth"] >= 4) & (num["elapsed"] <= 2 * num["soft"])
    soft, depth, lnodes = num["soft"][keep], num["depth"][keep], np.log1p(num["nodes"][keep])

    edges = [0, 150, 300, 600, 1200, 1e9]
    names = ["<150 ms", "150-300", "300-600", "600-1200", ">1200"]
    cols = [("depth (absolu)", depth), ("log_nodes (absolu)", lnodes)] + \
           [(n, X[:, i]) for i, n in enumerate(train_tm.FEATURES)]
    print(f"{'tranche':10} {'n':>8} {'cible':>7}")
    for lo, hi, nm in zip(edges[:-1], edges[1:], names):
        m = (soft >= lo) & (soft < hi)
        print(f"{nm:10} {m.sum():8d} {y[m].mean():7.4f}")
    for cname, v in cols:
        print(f"\n{cname}  (q10 / q50 / q90)")
        for lo, hi, nm in zip(edges[:-1], edges[1:], names):
            m = (soft >= lo) & (soft < hi)
            q = np.quantile(v[m], [0.1, 0.5, 0.9])
            print(f"  {nm:10} {q[0]:8.3f} {q[1]:8.3f} {q[2]:8.3f}")


if __name__ == "__main__" and sys.argv[1] != "--relation":
    main()


def relation():
    """Taux de la cible par tranche d'iters_left, a chaque cadence : c'est la
    relation que le modele apprend qui doit etre invariante, pas forcement la
    distribution (un budget long contient plus d'iterations au debut)."""
    rows = []
    for path in sys.argv[2:]:
        with open(path) as f:
            rows += [l for l in f]
    tmp = "/tmp/_inv2.tsv"
    with open(tmp, "w") as f:
        f.writelines(rows)
    X, y, _, _ = train_tm.load(tmp)
    a = np.array([l.rstrip("\n").split("\t") for l in rows if len(l.split("\t")) == len(train_tm.COLS)], dtype=object)
    num = {c: a[:, i].astype(np.float64) for i, c in enumerate(train_tm.COLS) if c not in ("move", "final")}
    key = num["game"] * 1000 + num["ply"]
    last = np.r_[key[1:] != key[:-1], True]
    keep = ~last & (num["depth"] >= 4) & (num["elapsed"] <= 2 * num["soft"])
    soft = num["soft"][keep]
    il = X[:, train_tm.FEATURES.index("iters_left")]
    st = X[:, train_tm.FEATURES.index("stable")]
    sedges = [0, 150, 300, 600, 1200, 1e9]
    snames = ["<150", "150-300", "300-600", "600-1200", ">1200"]
    iedges = [-9, 0, 1, 2, 3, 4, 6, 99]
    print("\nP(cible) par iters_left (lignes) et budget (colonnes), n entre parentheses")
    print(" " * 10 + "".join(f"{n:>16}" for n in snames))
    for lo, hi in zip(iedges[:-1], iedges[1:]):
        line = f"[{lo:>2},{hi:>2}) "
        for slo, shi in zip(sedges[:-1], sedges[1:]):
            m = (il >= lo) & (il < hi) & (soft >= slo) & (soft < shi)
            line += f"{y[m].mean():9.3f} ({m.sum():5d})" if m.sum() > 50 else f"{'-':>16}"
        print(line)
    print("\nMediane de stable dans la zone de decision (iters_left < 2)")
    for slo, shi, n in zip(sedges[:-1], sedges[1:], snames):
        m = (il < 2) & (soft >= slo) & (soft < shi)
        print(f"  {n:10} {np.median(st[m]):6.1f}  P(cible)={y[m].mean():.3f}")


if __name__ == "__main__" and sys.argv[1] == "--relation":
    relation()
