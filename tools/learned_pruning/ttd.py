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
            if line.startswith("info string oracle"):
                t = line.split()
                for k, v in zip(t[3::2], t[4::2]):  # "info string oracle nodes N ..."
                    oracle[k] = oracle.get(k, 0) + int(v)
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
    if o:  # oracle des coups calmes tardifs (ALCYON_ORACLE_QUIET, build d'experimentation)
        print(f"  oracle : retirable {o['removed_all'] / o['nodes']:.1%} des noeuds (non-PV seuls {o['removed_nonpv'] / o['nodes']:.1%}) ; "
              f"coups calmes tardifs inutiles {o['useless'] / o['late']:.2%} de {o['late']}")


if __name__ == "__main__":
    main()
