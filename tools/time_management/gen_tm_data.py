"""Genere le jeu de donnees de la gestion du temps apprise.

Rejoue des parties (PGN) coup par coup avec un moteur par processus, lance
une recherche longue (movetime) sur chaque position et recupere les lignes
"TM" que le moteur ecrit a chaque iteration terminee (ALCYON_TM_LOG=1).

Sortie : une ligne par iteration, colonnes
  game ply depth elapsed soft nodes stable frac_best frac_second score
  prev_score root_moves seldepth pieces move final_move
La cible se deduit ensuite : le coup de l'iteration differe-t-il du coup
final ?

Usage :
  python gen_tm_data.py ENGINE OUT.tsv PGN [PGN...] --workers 60 --movetime 3000
"""

import argparse
import multiprocessing as mp
import os
import subprocess
import math
import random
import time
import sys

import chess
import numpy as np
import chess.pgn


def load_games(paths, min_ply, every):
    games = []
    for path in paths:
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                board = g.board()
                start_fen = board.fen()
                moves = [m.uci() for m in g.mainline_moves()]
                if len(moves) > min_ply:
                    games.append((start_fen, moves))
    return [(i, fen, moves, min_ply, every) for i, (fen, moves) in enumerate(games)]


class Engine:
    def __init__(self, path):
        env = dict(os.environ, ALCYON_TM_LOG="1")
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.send("uci")
        self.wait("uciok")
        for opt in ("Threads value 1", "Hash value 16", "Ponder value false", "OwnBook value false"):
            self.send("setoption name " + opt)
        self.send("isready")
        self.wait("readyok")

    def send(self, cmd):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def wait(self, token):
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine died")
            if line.startswith(token):
                return line

    def search(self, fen, moves, movetime):
        pos = "position fen " + fen + (" moves " + " ".join(moves) if moves else "")
        self.send(pos)
        self.send(f"go movetime {movetime}")
        self.wait("bestmove")
        # Les lignes TM sont sur stderr ; on les lit jusqu'au marqueur.
        self.send("isready")
        self.wait("readyok")


def soft_budget(clock, inc, overhead=100, divisor=28, inc_frac=50):
    # Meme formule que uci.hpp (sans movestogo).
    return max(1, clock - overhead) // divisor + inc * inc_frac // 100


def worker(args):
    engine_path, movetime, jobs, out_path, mixed = args
    rng = random.Random(hash(out_path) & 0xffffffff)
    eng = Engine(engine_path)
    # stderr lu par un thread pour ne pas bloquer le moteur.
    import queue
    import threading
    q = queue.Queue()

    def pump():
        for line in eng.p.stderr:
            if line.startswith("TM "):
                q.put(line.split()[1:])
            elif line.startswith("L0 "):
                q.put(("L0", line[3:].strip()))
    threading.Thread(target=pump, daemon=True).start()

    l0_out = open(out_path + ".l0", "wb") if os.environ.get("ALCYON_TM_L0") else None
    with open(out_path, "w") as out:
        for gid, fen, moves, min_ply, every in jobs:
            eng.send("ucinewgame")
            eng.send("isready")
            eng.wait("readyok")
            for ply in range(min_ply, len(moves) - 1, every):
                soft = 0
                mt = movetime
                if mixed:
                    # Cadence tiree log-uniforme entre 5+0.05 et 60+0.6, horloge
                    # restante entre 15 % et 100 % de la base : budget souple
                    # reel, et recherche 3x plus longue pour connaitre la suite.
                    base = math.exp(rng.uniform(math.log(5000), math.log(60000)))
                    inc = base / 100
                    clock = base * rng.uniform(0.15, 1.0)
                    soft = soft_budget(int(clock), int(inc))
                    mt = min(8000, 3 * soft + 100)
                eng.search(fen, moves[:ply], mt)
                time.sleep(0.02)
                rows, l0 = [], None
                while not q.empty():
                    item = q.get()
                    if isinstance(item, tuple):
                        l0 = item[1]
                    else:
                        rows.append(item)
                if not rows:
                    continue
                if l0 is not None and l0_out is not None:
                    l0_out.write(np.array([gid, ply], dtype=np.int32).tobytes() + bytes.fromhex(l0))
                final = rows[-1][-1]
                for r in rows:
                    if mixed:
                        r[2] = str(soft)
                    out.write("\t".join([str(gid), str(ply)] + r + [final]) + "\n")
            out.flush()
    if l0_out is not None:
        l0_out.close()
    eng.send("quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("out")
    ap.add_argument("pgn", nargs="+")
    ap.add_argument("--workers", type=int, default=60)
    ap.add_argument("--movetime", type=int, default=3000)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--mixed", action="store_true", help="cadences melangees, budget souple journalise")
    a = ap.parse_args()

    jobs = load_games(a.pgn, a.min_ply, a.every)
    if a.max_games:
        jobs = jobs[:a.max_games]
    print(f"{len(jobs)} parties", file=sys.stderr)
    shards = [jobs[i::a.workers] for i in range(a.workers)]
    parts = [f"{a.out}.part{i}" for i in range(a.workers)]
    with mp.Pool(a.workers) as pool:
        pool.map(worker, [(a.engine, a.movetime, s, p, a.mixed) for s, p in zip(shards, parts)])
    with open(a.out, "w") as out:
        for p in parts:
            with open(p) as f:
                out.write(f.read())
            os.remove(p)
    if os.environ.get("ALCYON_TM_L0"):
        # Un enregistrement par position : int32 partie, int32 ply, 1024 octets.
        with open(a.out + ".l0", "wb") as out:
            for p in parts:
                with open(p + ".l0", "rb") as f:
                    out.write(f.read())
                os.remove(p + ".l0")


if __name__ == "__main__":
    main()
