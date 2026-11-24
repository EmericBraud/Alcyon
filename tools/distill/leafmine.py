"""Feuilles de variante a partir d'un diagnostic existant (run1).
Pour chaque coup d'Alcyon ou son score et celui de SF divergent d'au moins --min-gap
(unites internes), on relance Alcyon (meme profondeur) pour retrouver sa PV, on joue la PV
jusqu'a la feuille, et on enregistre : eval statique d'Alcyon + score SF (profondeur fixe).
Sortie au meme format que les .leaf.gz de gen2.py (resultat de partie inclus), par morceaux.
Usage : python3 leafmine.py --inp ~/gen/run1 --out ~/gen/run1_leaves --workers 64 --min-gap 200
"""
import argparse, glob, gzip, os, subprocess, time
from multiprocessing import Process
import chess, chess.engine

HOME = os.path.expanduser("~")


def load(inp, gap):
    tasks = []
    for f in sorted(glob.glob(os.path.join(inp, "*.diag.gz"))):
        for l in gzip.open(f, "rt"):
            fen, mv, a_sc, sf_sc, ply, a_res = l.rstrip("\n").split("\t")
            if a_sc == "None":
                continue
            a_sc, sf_sc = int(a_sc), int(sf_sc)
            if abs(a_sc) >= 30000 or abs(sf_sc) >= 30000 or abs(a_sc - sf_sc) < gap:
                continue
            tasks.append((fen, mv, a_sc, sf_sc, int(a_res)))
    return tasks


def worker(wid, a, tasks):
    env = dict(os.environ, ALCYON_DATA_DIR=f"{HOME}/v62/build/data")
    alc = chess.engine.SimpleEngine.popen_uci(f"{HOME}/v62/build/alcyon", env=env, timeout=600)
    alc.configure({"Hash": 16, "Threads": 1})
    sf = chess.engine.SimpleEngine.popen_uci(f"{HOME}/sfraw/src/stockfish", timeout=600)
    sf.configure({"Threads": 1, "Hash": 16})
    ev = subprocess.Popen([f"{HOME}/v62/build/alcyon"], env=env, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def static_eval(fen):
        ev.stdin.write(f"position fen {fen}\neval\n")
        ev.stdin.flush()
        while True:
            l = ev.stdout.readline()
            if not l:
                return None
            if l.startswith("score:"):
                return int(l.split()[1])

    chunk, out, tag, n_in_chunk, done = 0, None, None, 0, 0
    t0 = time.time()
    stat = open(os.path.join(a.out, f"m{wid:03d}.stat"), "w")

    def rotate():
        nonlocal chunk, out, tag, n_in_chunk
        if out:
            out.close()
            os.rename(os.path.join(a.out, tag + ".leaf.gz.tmp"), os.path.join(a.out, tag + ".leaf.gz"))
        chunk += 1
        tag = f"m{wid:03d}_{chunk:05d}"
        out = gzip.open(os.path.join(a.out, tag + ".leaf.gz.tmp"), "wt")
        n_in_chunk = 0

    rotate()
    for fen, mv, a_sc0, sf_sc, a_res in tasks:
        board = chess.Board(fen)
        alc_turn = board.turn
        r = alc.analyse(board, chess.engine.Limit(depth=a.alc_depth))
        pv = r.get("pv") or []
        a_sc = r["score"].relative.score(mate_score=32000) if "score" in r else a_sc0
        lb = board.copy()
        n_pv = 0
        for m in pv:
            if not lb.is_legal(m):
                break
            lb.push(m)
            n_pv += 1
        if n_pv >= 1 and not lb.is_check() and not lb.is_game_over():
            st_ev = static_eval(lb.fen())
            li = sf.analyse(lb, chess.engine.Limit(depth=a.sf_depth))
            ls = li["score"].relative
            if st_ev is not None and not ls.is_mate():
                r_stm = a_res if lb.turn == alc_turn else -a_res
                out.write(f"{lb.fen()}\t{li['pv'][0].uci()}\t{ls.score()}\t{n_pv}\t{st_ev}\t{a_sc}\t{r_stm}\n")
                n_in_chunk += 1
        done += 1
        if n_in_chunk >= a.per_chunk:
            rotate()
        if done % 200 == 0:
            stat.seek(0); stat.write(f"{done} {len(tasks)} {time.time() - t0:.0f}\n"); stat.flush()
    out.close()
    os.rename(os.path.join(a.out, tag + ".leaf.gz.tmp"), os.path.join(a.out, tag + ".leaf.gz"))
    stat.seek(0); stat.write(f"{done} {len(tasks)} {time.time() - t0:.0f}\n"); stat.flush()
    alc.quit(); sf.quit(); ev.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--min-gap", type=int, default=200)
    ap.add_argument("--alc-depth", type=int, default=8)
    ap.add_argument("--sf-depth", type=int, default=9)
    ap.add_argument("--per-chunk", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = load(a.inp, a.min_gap)
    if a.limit:
        tasks = tasks[: a.limit]
    print(len(tasks), "positions a traiter", flush=True)
    ps = [Process(target=worker, args=(i, a, tasks[i::a.workers])) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("termine", flush=True)
