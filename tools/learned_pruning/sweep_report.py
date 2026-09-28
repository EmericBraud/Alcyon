"""Agrege sweep.sh : par config, noeuds totaux et accord du meilleur coup.

    python3 sweep_report.py <dossier de sortie de sweep.sh>

Accord avec ref14 (mecanisme eteint, profondeur 14) : une config vaut
quelque chose si, a noeuds egaux, elle s'accorde plus souvent avec ref14
que la courbe off11 -> off12 -> off13 (mecanisme eteint).
"""
import collections, glob, os, re, sys

LINE = re.compile(r"info string bench (\d+)/\d+ nodes (\d+) .* bestmove (\S+)")
out = sys.argv[1]
moves = collections.defaultdict(dict)   # config -> (lot, i) -> coup
nodes = collections.Counter()
for path in glob.glob(os.path.join(out, "*__lot_*.log")):
    name, lot = os.path.basename(path)[:-4].split("__")
    for m in LINE.finditer(open(path).read()):
        moves[name][(lot, int(m[1]))] = m[3]
        nodes[name] += int(m[2])
ref, base = moves["ref14"], moves["off12"]
keys = sorted(set(ref) & set.intersection(*(set(v) for v in moves.values())))
print(f"{len(keys)} positions communes a toutes les configs\n")
print(f"{'config':14s} {'noeuds/off12':>12s} {'accord ref14':>13s} {'accord off12':>13s}")
for name in sorted(moves, key=lambda n: nodes[n]):
    a_ref = sum(moves[name][k] == ref[k] for k in keys) / len(keys)
    a_base = sum(moves[name][k] == base[k] for k in keys) / len(keys)
    print(f"{name:14s} {nodes[name] / nodes['off12']:12.3f} {a_ref:13.1%} {a_base:13.1%}")
