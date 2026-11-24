"""Rejuge, a budget SF eleve, tous les coups candidats de rethink.json :
coup joue en partie, meilleur coup SF (300k), coups d'Alcyon a 1/4/16/64 s.
Par position : ES de la meilleure suite (SF analyse la racine) et ES apres chaque coup.
Usage : python rejudge.py rethink.json out.json --sf SF --nodes 10000000 --workers 100"""
import argparse, json, os, time
from multiprocessing import Pool
import chess, chess.engine

SF = None
NODES = 0


def init(sf, nodes):
    global SF, NODES
    time.sleep((os.getpid() % 100) * 0.3)
    SF = chess.engine.SimpleEngine.popen_uci(sf, timeout=900)
    SF.configure({"Threads": 1, "Hash": 256, "UCI_ShowWDL": True})
    NODES = nodes


def es(info, pov):
    w = info["wdl"].pov(pov)
    return (w.wins + w.draws / 2) / 1000


def job(r):
    b = chess.Board(r["fen"])
    pov = b.turn
    lim = chess.engine.Limit(nodes=NODES)
    root = SF.analyse(b, lim)
    out = {"fen": r["fen"], "best_hi": root["pv"][0].uci(), "root_es": es(root, pov), "moves": {}}
    cands = {r["move"], r["best"]} | {x["move"] for x in r["runs"]}
    for u in cands:
        if not u:
            continue
        bb = b.copy(); bb.push(chess.Move.from_uci(u))
        if bb.is_game_over():
            res = bb.result()
            v = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == (pov == chess.WHITE) else 0.0)
        else:
            v = es(SF.analyse(bb, lim), pov)
        out["moves"][u] = v
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--sf", required=True)
    ap.add_argument("--nodes", type=int, default=10000000)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    R = json.load(open(a.inp))
    if a.limit:
        R = R[: a.limit]
    res = []
    with Pool(a.workers, init, (a.sf, a.nodes)) as pool:
        for i, o in enumerate(pool.imap_unordered(job, R)):
            res.append(o)
            if i % 20 == 0:
                print(i, flush=True)
                json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"))
    print("termine", flush=True)
