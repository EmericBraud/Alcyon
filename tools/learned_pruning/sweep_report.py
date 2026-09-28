"""Agrege sweep.sh : par config, noeuds totaux et accord du meilleur coup.

    python3 sweep_report.py <dossier de sortie de sweep.sh> [ref_sf.txt | config]

Reference par defaut : ref14. Une config du balayage peut servir de
reference (ex. ref15_s1 : Alcyon mono-thread, profondeur 15, bruit
d'ordonnancement graine 1 -- decorrelee de off11..off13, qui sinon sont
litteralement le debut de l'arbre de la reference).

Avec ref_sf.txt (sf_ref.py), l'accord est mesure contre Stockfish au lieu
de ref14 -- ref14 partage le debut de son arbre avec off11..off13.

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
# Reference independante optionnelle (sf_ref.py) : index de ligne du fichier
# de positions -> coup ; sweep.sh decoupe en lots de 25, dans l'ordre.
REF = "ref14"
if len(sys.argv) > 2 and not os.path.exists(sys.argv[2]):
    REF = sys.argv[2]  # une config du balayage, ex. ref15_s1
elif len(sys.argv) > 2:
    REF = "sf"
    moves["sf"] = {}
    for line in open(sys.argv[2]):
        n, mv = line.split()
        moves["sf"][(f"lot_{int(n) // 25:04d}", int(n) % 25 + 1)] = mv
    nodes["sf"] = nodes["off12"]
ref, base = moves[REF], moves["off12"]
print(f"reference : {REF}")
keys = sorted(set(ref) & set.intersection(*(set(v) for v in moves.values())))
print(f"{len(keys)} positions communes a toutes les configs\n")
print(f"{'config':14s} {'noeuds/off12':>12s} {'accord ref':>13s} {'accord off12':>13s}")
for name in sorted(moves, key=lambda n: nodes[n]):
    a_ref = sum(moves[name][k] == ref[k] for k in keys) / len(keys)
    a_base = sum(moves[name][k] == base[k] for k in keys) / len(keys)
    print(f"{name:14s} {nodes[name] / nodes['off12']:12.3f} {a_ref:13.1%} {a_base:13.1%}")
