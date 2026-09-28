"""Ecart apparie config - base a noeuds egaux (isonode.sh), avec IC 95 %.

    [PREFIX=iso] python3 isonode_report.py <dossier de sweep.sh> <reference> <config> [config...]

Pour chaque position : la config et la base (meme budget de noeuds)
s'accordent-elles avec la reference ? Ecart = moyenne des differences,
position par position ; IC 95 % par l'erreur type de cette difference.
Pas d'interpolation de courbe, pas de biais de concavite.
"""
import glob
import os
import re
import sys

import numpy as np

LINE = re.compile(r"info string bench (\d+)/\d+ nodes (\d+) .* bestmove (\S+)")


def moves(out, name):
    m, n = {}, 0
    for path in glob.glob(os.path.join(out, f"{name}__lot_*.log")):
        lot = os.path.basename(path)[:-4].split("__")[1]
        for g in LINE.finditer(open(path).read()):
            m[(lot, int(g[1]))] = g[3]
            n += int(g[2])
    return m, n


def main():
    out, ref_name, configs = sys.argv[1], sys.argv[2], sys.argv[3:]
    ref, _ = moves(out, ref_name)
    off, off_nodes = moves(out, "off12")
    print(f"reference : {ref_name}\n")
    print(f"{'config':16s} {'noeuds/off12':>12s} {'accord':>7s} {'base iso':>9s} {'ecart':>7s} {'IC 95 %':>16s} {'positions':>9s}")
    for cfg in configs:
        c, c_nodes = moves(out, cfg)
        b, _ = moves(out, os.environ.get("PREFIX", "iso") + "_" + cfg)
        keys = sorted(set(c) & set(b) & set(ref) & set(off))
        a_c = np.array([c[k] == ref[k] for k in keys], float)
        a_b = np.array([b[k] == ref[k] for k in keys], float)
        d = a_c - a_b
        se = d.std(ddof=1) / np.sqrt(len(d))
        ratio = c_nodes / off_nodes
        print(f"{cfg:16s} {ratio:12.3f} {a_c.mean():7.1%} {a_b.mean():9.1%} {d.mean():+7.2%} "
              f"[{d.mean() - 1.96 * se:+6.2%}, {d.mean() + 1.96 * se:+6.2%}] {len(keys):9d}")


if __name__ == "__main__":
    main()
