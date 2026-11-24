"""Rejoue Alcyon sur ses positions d'erreur avec plus de temps ; SF juge le nouveau coup.
Aussi : eval statique / recherche courte d'Alcyon sur le coup joue vs le meilleur coup SF.
Usage : python rethink.py quiet_err.json out.json --alc RUN --sf SF --times 1000,4000,16000 --workers 100"""
import argparse, json, os, time
from multiprocessing import Pool
import chess, chess.engine

AL = SF = None
ARGS = None


def init(alc, sf):
    global AL, SF
    time.sleep((os.getpid() % 100) * 0.3)
    AL = chess.engine.SimpleEngine.popen_uci(alc, timeout=900)
    AL.configure({"Threads": 1, "Hash": 64})
    SF = chess.engine.SimpleEngine.popen_uci(sf, timeout=900)
    SF.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})


def es(info, pov):
    w = info.get("wdl")
    if w is not None:
        w = w.pov(pov)
        return (w.wins + w.draws / 2) / 1000
    return 0.5


def judge(b, mv, pov, cache):
    if mv in cache:
        return cache[mv]
    bb = b.copy()
    bb.push(mv)
    if bb.is_game_over():
        r = bb.result()
        v = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == (pov == chess.WHITE) else 0.0)
    else:
        v = es(SF.analyse(bb, chess.engine.Limit(nodes=300000)), pov)
    cache[mv] = v
    return v


def job(task):
    p, times = task
    b = chess.Board(p["fen"])
    pov = b.turn
    best_es = es(SF.analyse(b, chess.engine.Limit(nodes=300000)), pov)
    cache = {}
    out = dict(p, best_es=best_es, runs=[])
    played = chess.Move.from_uci(p["move"])
    sfbest = chess.Move.from_uci(p["best"]) if p["best"] else None
    # Alcyon : score de son coup joue et du meilleur coup SF, recherche courte (depth 12) apres chaque coup
    for name, mv in [("played", played), ("sfbest", sfbest)]:
        if mv is None:
            continue
        bb = b.copy(); bb.push(mv)
        i = AL.analyse(bb, chess.engine.Limit(depth=12))
        out["alc_" + name] = -i["score"].pov(not pov).score(mate_score=100000) if False else i["score"].pov(pov).score(mate_score=100000)
    for t in times:
        AL.protocol.send_line("ucinewgame") if False else None
        r = AL.play(b, chess.engine.Limit(time=t / 1000), info=chess.engine.INFO_ALL)
        mv = r.move
        out["runs"].append(dict(t=t, move=mv.uci(), es=judge(b, mv, pov, cache),
                                depth=r.info.get("depth"), score=r.info["score"].pov(pov).score(mate_score=100000) if "score" in r.info else None))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--alc", required=True); ap.add_argument("--sf", required=True)
    ap.add_argument("--times", default="1000,4000,16000")
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ps = json.load(open(a.inp))
    if a.limit:
        ps = ps[: a.limit]
    times = [int(x) for x in a.times.split(",")]
    res = []
    with Pool(a.workers, init, (a.alc, a.sf)) as pool:
        for i, r in enumerate(pool.imap_unordered(job, [(p, times) for p in ps])):
            res.append(r)
            if i % 20 == 0:
                print(i, flush=True)
                json.dump(res, open(a.out, "w"))
    json.dump(res, open(a.out, "w"))
    print("termine", flush=True)
