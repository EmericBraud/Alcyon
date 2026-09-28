"""Compare les predictions de train_mlp.py --out et la regression integree.

    python3 compare_mlp.py <test.bin> <learned_prune_weights.hpp> pred_0.npy pred_16.npy ...

"deploye" = la regression par profondeur reellement dans le moteur
(learned_prune_weights.hpp, convergee par scikit-learn) : c'est le modele a
battre. Les predictions doivent venir du meme test.bin (meme filtre).
"""
import os
import re
import sys

import numpy as np

import fit
import train_mlp

test_path, header, preds = sys.argv[1], sys.argv[2], sys.argv[3:]
r, _, idx = train_mlp.load(test_path)
rt = r[idx]
body = open(header).read()
W = np.array([float(v) for v in re.findall(r"(-?\d\.\d+e[+-]\d+)f", body[body.index("kWeights"):])]).reshape(len(fit.SLICES), -1)
m, x = fit.features(rt)
X = fit.design(m, x, "b")
s = np.minimum(rt["depth"], len(fit.SLICES)) - 1
models = {"deploye": 1 / (1 + np.exp(-(W[s, 0] + np.einsum("ij,ij->i", X, W[s, 1:]))))}
for p in preds:
    models[os.path.basename(p).replace("pred_", "h").replace(".npy", "")] = np.load(p)
y, sub, dep = rt["outcome"] != 1, rt["subtree"], rt["depth"]
print(f"{len(rt)} noeuds de test")
print(f"{'depth':>6s} {'modele':>8s} {'logloss':>8s}  part du sous-arbre coupable a precision >= 98 / 99 / 99.5 %")
for lo, hi in fit.SLICES:
    k = (dep >= lo) & (dep <= hi)
    if not k.any():
        continue
    for name, p in models.items():
        ps = np.clip(p[k], 1e-7, 1 - 1e-7)
        ll = -np.mean(y[k] * np.log(ps) + (~y[k]) * np.log(1 - ps))
        cov = " / ".join(f"{fit.coverage(p[k], y[k], sub[k], t)[1]:5.1%}" for t in (0.98, 0.99, 0.995))
        print(f"{(str(lo) if lo == hi else f'{lo}+'):>6s} {name:>8s} {ll:8.4f}  {cov}")
