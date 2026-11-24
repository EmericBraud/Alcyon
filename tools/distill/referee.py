"""Arbitrage des coups d'Alcyon par Stockfish actuel (noeuds fixes).
Pour chaque coup d'Alcyon : eval SF avant (meilleur coup) et apres le coup joue,
en score attendu (WDL SF) et en cp. Sortie CSV.
Usage : python referee.py games.pgn out.csv --sf PATH --nodes 500000 --workers 32"""
import argparse, csv, re, sys
from multiprocessing import Pool
import chess, chess.engine, chess.pgn

CMT = re.compile(r"/(\d+) ([\d.]+)s")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
A = None


def init(sf):
    global A
    import os, time
    time.sleep((os.getpid() % 100) * 0.3)
    A = chess.engine.SimpleEngine.popen_uci(sf, timeout=900)
    A.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})


def es(info, pov):
    s = info["score"].pov(pov)
    w = info.get("wdl")
    if w is not None:
        w = w.pov(pov) if hasattr(w, "pov") else w
        try:
            return (w.wins + w.draws / 2) / 1000, s.score(mate_score=10000)
        except AttributeError:
            return (w[0] + w[1] / 2) / 1000, s.score(mate_score=10000)
    cp = s.score(mate_score=10000)
    return 1 / (1 + 10 ** (-cp / 400)), cp


def job(task):
    gi, ply, fen, uci, a_white, depth, t, mat, nodes = task
    b = chess.Board(fen)
    pov = b.turn
    A.protocol.send_line("ucinewgame") if False else None
    lim = chess.engine.Limit(nodes=nodes)
    i0 = A.analyse(b, lim)
    es0, cp0 = es(i0, pov)
    best = i0.get("pv", [None])[0]
    mv = chess.Move.from_uci(uci)
    if best == mv:
        es1, cp1 = es0, cp0
    else:
        b.push(mv)
        if b.is_game_over():
            r = b.result()
            es1 = 1.0 if (r == "1-0") == (pov == chess.WHITE) and r != "1/2-1/2" else (0.5 if r == "1/2-1/2" else 0.0)
            cp1 = {1.0: 10000, 0.5: 0, 0.0: -10000}[es1]
        else:
            i1 = A.analyse(b, lim)
            es1, cp1 = es(i1, pov)
    return [gi, ply, int(a_white), depth, t, mat, round(es0, 4), round(es1, 4), cp0, cp1, uci, best.uci() if best else ""]


def tasks(pgn, nodes):
    with open(pgn) as f:
        gi = 0
        while (g := chess.pgn.read_game(f)):
            aw = g.headers["White"].startswith("alcyon")
            b = g.board()
            for node in g.mainline():
                if (b.turn == chess.WHITE) == aw:
                    m = CMT.search(node.comment or "")
                    d, t = (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)
                    mat = sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values())
                    yield (gi, b.ply(), b.fen(), node.move.uci(), aw, d, t, mat, nodes)
                b.push(node.move)
            gi += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn"); ap.add_argument("out")
    ap.add_argument("--sf", required=True)
    ap.add_argument("--nodes", type=int, default=500000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    ts = list(tasks(a.pgn, a.nodes))
    if a.limit:
        ts = ts[: a.limit]
    print(len(ts), "coups", flush=True)
    with open(a.out, "w", newline="") as f, Pool(a.workers, init, (a.sf,)) as p:
        w = csv.writer(f)
        w.writerow(["game", "ply", "alc_white", "depth", "time", "mat", "es_before", "es_after", "cp_before", "cp_after", "move", "best"])
        for i, r in enumerate(p.imap_unordered(job, ts, chunksize=8)):
            w.writerow(r)
            if i % 1000 == 0:
                print(i, flush=True); f.flush()
    print("termine", flush=True)
