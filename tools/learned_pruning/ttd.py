"""Cout des sondes en conditions de partie : noeuds et temps a profondeur egale.

    python3 ttd.py <binaire> <dossier ctx> <profondeur> <nom> [option=valeur ...]

Rejoue chaque partie de <dossier>/game_*.txt (context_bias.py prep) coup par
coup, TT conservee, Hash = 8, "go depth D". Une partie par processus, toutes
en parallele. Imprime, sommes sur toutes les parties : coups, noeuds, temps
moteur (ms, "info ... time"), et profondeur selective moyenne.
"""
import glob
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def run(engine, game_file, depth, opts):
    lines = open(game_file).read().split()
    ply_min, moves = int(lines[0]), lines[1:]
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    setopts = "".join(f"setoption name {k} value {v}\n" for k, v in opts)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name Hash value 8\n"
                  f"setoption name OwnBook value false\n{setopts}isready\n")
    while p.stdout.readline().strip() != "readyok":
        pass
    p.stdin.write("ucinewgame\n")
    n = nodes = ms = 0
    oracle = {}
    for ply in range(ply_min, len(moves)):
        p.stdin.write(f"position startpos moves {' '.join(moves[:ply])}\ngo depth {depth}\n")
        last = None
        while True:
            line = p.stdout.readline()
            if not line:
                sys.exit(f"{game_file} : moteur mort au ply {ply}")
            if line.startswith("info string oracle") or line.startswith("info string oorder"):
                t = line.split()  # "info string <tag> cle valeur ..."
                for k, v in zip(t[3::2], t[4::2]):
                    oracle[t[2] + ":" + k] = oracle.get(t[2] + ":" + k, 0) + int(v)
            elif line.startswith("info") and " nodes " in line:
                last = line.split()
            if line.startswith("bestmove"):
                break
        if last:
            nodes += int(last[last.index("nodes") + 1])
            ms += int(last[last.index("time") + 1]) if "time" in last else 0
            n += 1
    p.stdin.write("quit\n")
    p.wait()
    return n, nodes, ms, oracle


def main():
    engine, folder, depth, name = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    opts = [a.split("=", 1) for a in sys.argv[5:]]
    games = sorted(glob.glob(os.path.join(folder, "game_*.txt")))
    with ThreadPoolExecutor(int(os.environ.get("TTD_JOBS", os.cpu_count()))) as ex:
        res = list(ex.map(lambda g: run(engine, g, depth, opts), games))
    n, nodes, ms = (sum(r[i] for r in res) for i in range(3))
    print(f"{name:12s} coups {n}  noeuds {nodes}  temps {ms} ms  noeuds/coup {nodes / n:.0f}  ms/coup {ms / n:.1f}")
    o = {}
    for r in res:
        for k, v in r[3].items():
            o[k] = o.get(k, 0) + v
    if "oracle:nodes" in o:  # oracle des coups calmes tardifs (ALCYON_ORACLE_QUIET)
        g = lambda k: o["oracle:" + k]  # noqa: E731
        print(f"  oracle : retirable {g('removed_all') / g('nodes'):.1%} des noeuds (non-PV seuls {g('removed_nonpv') / g('nodes'):.1%}) ; "
              f"coups calmes tardifs inutiles {g('useless') / g('late'):.2%} de {g('late')}")
    if "oorder:nodes" in o:  # oracle d'ordonnancement (ALCYON_ORACLE_ORDER)
        g = lambda k: o["oorder:" + k]  # noqa: E731
        names = ["TT", "promotions", "bonnes prises", "killers", "contre-coup", "calmes", "mauvaises prises", "?"]
        print(f"  ordre : retirable {g('removed') / g('nodes'):.1%} des noeuds ; coupures au 1er coup {g('first') / g('cut'):.1%} de {g('cut')}")
        wt = sum(g(f"waste_s{i}") for i in range(8))
        for i in range(7):
            if g(f"cut_s{i}"):
                print(f"    coupures par {names[i]:17s} {g(f'cut_s{i}') / g('cut'):6.1%} ; gaspillage avant elles {g(f'waste_s{i}') / wt:6.1%}")


if __name__ == "__main__":
    main()
