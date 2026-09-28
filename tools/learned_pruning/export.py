"""Entraine le modele (b) de fit.py par profondeur et l'ecrit en header C++.

    python3 export.py train.bin src/engine/search/learned_prune_weights.hpp
    python3 export.py check dump.bin src/engine/search/learned_prune_weights.hpp

`check` relit un dump v2 (qui contient le logit calcule par le moteur) et le
compare au logit recalcule ici depuis les memes poids : c'est la preuve que
learned_prune.hpp reconstruit bien les features de fit.py.

Tous les noeuds sont gardes, NMP compris : le mecanisme est place AVANT le
NMP, il voit donc exactement cette distribution. La normalisation
(StandardScaler) est repliee dans les poids : z = b + sum(w_i * f_i), ou f
est le vecteur de fit.design(..., "b") -- learned_prune.hpp doit le
reconstruire a l'identique.
"""
import multiprocessing
import os
import sys

N_JOBS = 10
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(max(1, os.cpu_count() // N_JOBS)))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import fit

_TR = None


def train_slice(sl):
    lo, hi = sl
    r = _TR[(_TR["depth"] >= lo) & (_TR["depth"] <= hi)]
    if len(r) > fit.MAX_TRAIN:
        r = r[np.random.default_rng(0).choice(len(r), fit.MAX_TRAIN, replace=False)]
    m, x = fit.features(r)
    X = fit.design(m, x, "b").astype(np.float64)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), r["outcome"] != 1)
    w = clf.coef_[0] / sc.scale_
    b = clf.intercept_[0] - np.dot(clf.coef_[0], sc.mean_ / sc.scale_)
    return b, w


def main(train_path, out_path):
    global _TR
    _TR = fit.load(train_path)
    assert len(fit.SLICES) == N_JOBS
    with multiprocessing.get_context("fork").Pool(N_JOBS) as pool:
        res = pool.map(train_slice, fit.SLICES)
    n = len(res[0][1])
    with open(out_path, "w") as f:
        f.write("// Genere par tools/learned_pruning/export.py -- ne pas editer.\n")
        f.write(f"// Source : {os.path.basename(train_path)}, {len(_TR)} noeuds.\n")
        f.write("#pragma once\n\nnamespace learned_prune\n{\n")
        f.write(f"    constexpr int kSlices = {len(res)}; // depth 1..{len(res) - 1}, puis {len(res)}+\n")
        f.write(f"    constexpr int kInputs = {n};\n")
        f.write("    // kWeights[s][0] = biais, puis un poids par entree de fit.design(..., \"b\").\n")
        f.write("    constexpr float kWeights[kSlices][kInputs + 1] = {\n")
        for b, w in res:
            f.write("        {" + ", ".join(f"{v:.8e}f" for v in [b, *w]) + "},\n")
        f.write("    };\n}\n")
    print(f"ecrit {out_path} : {len(res)} tranches x {n + 1} poids")


def check(dump_path, header_path):
    import re
    txt = open(header_path).read()
    body = txt[txt.index("kWeights"):]
    vals = np.array([float(v) for v in re.findall(r"(-?\d\.\d+e[+-]\d+)f", body)])
    W = vals.reshape(len(fit.SLICES), -1)
    r = fit.load(dump_path)
    m, x = fit.features(r)
    X = fit.design(m, x, "b")
    s = np.minimum(r["depth"], len(fit.SLICES)) - 1
    z = W[s, 0] + np.einsum("ij,ij->i", X, W[s, 1:])
    err = np.abs(z - r["learned_z"])
    print(f"{len(r)} noeuds, ecart max {err.max():.2e}, moyen {err.mean():.2e}")
    assert err.max() < 1e-3, "le moteur et fit.py ne calculent pas les memes features"


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(*sys.argv[2:4])
    else:
        main(*sys.argv[1:3])
