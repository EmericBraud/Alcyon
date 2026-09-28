"""Precision en situation des decisions du mecanisme, depuis un dump v2.

    python3 precision.py dump.bin [t_base t_depth]

Pour chaque profondeur : parmi les noeuds ou le mecanisme couperait
(z >= logit(T) -> fail-high predit, z <= -logit(T) -> fail-low predit),
quelle part a vraiment eu ce resultat. A comparer entre un dump
learned_prune_enabled=0 (arbre d'entrainement) et =2 (mode ombre).
"""
import sys
import numpy as np
import fit

r = fit.load(sys.argv[1])
tb, td = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (950, 5)
fh = r["outcome"] != 1
print(f"{len(r)} noeuds, T(d) = ({tb} + {td} * d) / 1000")
print(f"{'depth':>6s} {'noeuds':>9s} {'coupes':>7s} {'prec. FH':>9s} {'prec. FL':>9s} {'prec.':>7s} {'prev2 connue':>13s}")
for d in range(1, 11):
    s = (r["depth"] == d) if d < 10 else (r["depth"] >= 10)
    t = min(max(tb + td * d, 500), 999) / 1000
    zt = np.log(t / (1 - t))
    z = r["learned_z"][s]
    hi, lo = z >= zt, z <= -zt
    n = hi.sum() + lo.sum()
    p_hi = fh[s][hi].mean() if hi.any() else float("nan")
    p_lo = (~fh[s][lo]).mean() if lo.any() else float("nan")
    p = (fh[s][hi].sum() + (~fh[s][lo]).sum()) / n if n else float("nan")
    known = (r["eval_prev2"][s] != fit.K_EVAL_NONE).mean()
    print(f"{d:>6d} {s.sum():>9d} {n / s.sum():>7.1%} {p_hi:>9.2%} {p_lo:>9.2%} {p:>7.2%} {known:>13.1%}")
