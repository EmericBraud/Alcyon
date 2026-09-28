"""Rapport de budget_eval.sh : ecart apparie config - base, meme budget, IC 95 %.

    python3 budget_report.py <dossier> <reference> <config> [config...]
"""
import os
import sys

import numpy as np

from isonode_report import moves

out, ref_name, configs = sys.argv[1], sys.argv[2], sys.argv[3:]
ref, _ = moves(out, ref_name)
base, base_nodes = moves(out, "nb_base")
print(f"reference : {ref_name} ; base = mecanisme eteint, meme budget de noeuds\n")
print(f"{'config':16s} {'noeuds/base':>11s} {'accord':>7s} {'base':>7s} {'ecart':>7s} {'IC 95 %':>18s} {'positions':>9s}")
for cfg in configs:
    c, c_nodes = moves(out, "nb_" + cfg)
    keys = sorted(set(c) & set(base) & set(ref))
    a_c = np.array([c[k] == ref[k] for k in keys], float)
    a_b = np.array([base[k] == ref[k] for k in keys], float)
    d = a_c - a_b
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"{cfg:16s} {c_nodes / base_nodes:11.3f} {a_c.mean():7.1%} {a_b.mean():7.1%} {d.mean():+7.2%} "
          f"[{d.mean() - 1.96 * se:+6.2%}, {d.mean() + 1.96 * se:+6.2%}] {len(keys):9d}")
