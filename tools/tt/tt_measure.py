"""Mesure l'utilisation reelle de la TT en conditions de partie.

Rejoue des parties (PGN) coup par coup avec un binaire compile avec
-DALCYON_TT_STATS : un moteur par partie, pendule simulee (cadence donnee,
horloge qui decroit comme en partie), TT conservee tout au long de la partie.
A la sortie du moteur, ses compteurs TTSTATS sont additionnes.

Usage : python tt_measure.py ENGINE PGN --games 120 --workers 60 --tc 10+0.1 --hash 16
"""

import argparse
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict

import chess.pgn


def load(path, n):
    games = []
    with open(path) as f:
        while len(games) < n:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            moves = [m.uci() for m in g.mainline_moves()]
            if len(moves) > 20:
                games.append((g.board().fen(), moves))
    return games


def run(args):
    engine, fen, moves, base_ms, inc_ms, hash_mb = args
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1)

    def send(c):
        p.stdin.write(c + "\n")
        p.stdin.flush()

    def wait(tok):
        while True:
            line = p.stdout.readline()
            if not line or line.startswith(tok):
                return line

    send("uci"); wait("uciok")
    for o in (f"Hash value {hash_mb}", "Threads value 1", "Ponder value false", "OwnBook value false"):
        send("setoption name " + o)
    send("ucinewgame"); send("isready"); wait("readyok")
    clock = [base_ms, base_ms]
    for ply in range(8, len(moves)):
        side = ply % 2
        # Les deux camps sont joues par le meme moteur (meme TT), comme en
        # auto-jeu ; l'horloge baisse d'une estimation du temps consomme.
        send(f"position fen {fen} moves {' '.join(moves[:ply])}")
        send(f"go wtime {clock[0]} btime {clock[1]} winc {inc_ms} binc {inc_ms}")
        wait("bestmove")
        used = max(1, clock[side] // 28 + inc_ms // 2)
        clock[side] = max(100, clock[side] - used + inc_ms)
    send("quit")
    err = p.stderr.read()
    p.wait()
    out = defaultdict(lambda: None)
    for line in err.splitlines():
        if line.startswith("TTSTATS"):
            parts = line.split()
            out[parts[1]] = parts[2:]
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("pgn")
    ap.add_argument("--games", type=int, default=120)
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--tc", default="10+0.1")
    ap.add_argument("--hash", type=int, default=16)
    a = ap.parse_args()
    base, inc = a.tc.split("+")
    base_ms, inc_ms = int(float(base) * 1000), int(float(inc) * 1000)
    games = load(a.pgn, a.games)
    with mp.Pool(a.workers) as pool:
        res = pool.map(run, [(a.engine, f, m, base_ms, inc_ms, a.hash) for f, m in games])

    tot = defaultdict(lambda: defaultdict(int))
    for r in res:
        for key, vals in r.items():
            if key in ("probes", "stores"):
                # lignes "cle valeur cle valeur ..."
                toks = [key] + vals
                for i in range(0, len(toks) - 1, 2):
                    tot["main"][toks[i]] += int(toks[i + 1])
            else:
                for i, v in enumerate(vals):
                    tot[key][i] += int(v)
    m = tot["main"]
    print(f"parties {len(res)}  cadence {a.tc}  hash {a.hash} Mo")
    print(f"probes {m['probes']}  trouvees {m['hits'] / m['probes']:.3f}  coupures {m['cuts'] / m['probes']:.3f}")
    st = m["stores"]
    print(f"ecritures {st} : vide {m['empty'] / st:.3f}  meme cle {m['same'] / st:.3f}  "
          f"eviction {m['evict'] / st:.3f}  refusee {m['skip'] / st:.3f}  (evincees d'une ancienne generation : "
          f"{m['evict_old_gen'] / max(1, m['evict']):.3f})")
    sh = [tot["slot_hits"][i] for i in range(4)]
    print("lectures par emplacement du paquet : " + " ".join(f"{x / max(1, sum(sh)):.3f}" for x in sh))
    names = ["0 lecture", "1", "2-3", "4-7", "8-31", "32+"]
    dn = ["prof <=0", "1-2", "3-5", "6-9", "10+"]
    print("entrees evincees, par profondeur (lignes) et nombre de lectures avant eviction (colonnes) :")
    print(" " * 10 + "".join(f"{n:>11}" for n in names))
    for d in range(5):
        row = [tot[f"evicted_d{d}"][i] for i in range(6)]
        tot_row = max(1, sum(row))
        print(f"{dn[d]:10}" + "".join(f"{x / tot_row:11.3f}" for x in row) + f"   (n={sum(row)})")


if __name__ == "__main__":
    main()
