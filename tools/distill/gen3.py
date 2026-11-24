"""[gen3 : echantillonnage AVANT etiquetage -- 1 position sur --sample etiquetee a
--sf-depth (+ sa feuille), les autres coups de SF a --sf-fast-depth ; fin de partie testee sans
claim_draw (couteux).]
Donnees + diagnostic : parties Alcyon (profondeur fixe) contre Stockfish actuel
(profondeur fixe, scores en unites internes brutes), depuis des ouvertures UHO.

- Entrainement (.plain.gz) : chaque position ou Stockfish a le trait (hors echec),
  etiquetee par SA recherche de partie (score + coup joue) et le resultat de la partie.
  Ce sont les positions issues des coups d'Alcyon.
- Diagnostic (.diag.gz, TSV) : a chaque coup d'Alcyon, son eval (son point de vue),
  puis l'eval de Stockfish au coup suivant (point de vue d'Alcyon) : l'ecart mesure
  l'erreur de jugement.

Fichiers par morceaux de --games-per-chunk parties, ecrits en .tmp puis renommes une
fois complets (seuls les fichiers finis sont sauvegardes).
Usage : python3 gen2.py --out DIR --workers 64 --alc-depth 8 --sf-depth 9
"""
import argparse, gzip, os, random, subprocess, time
from multiprocessing import Process
import chess, chess.engine

HOME = os.path.expanduser("~")


def worker(wid, a, openings):
    rnd = random.Random(f"{wid}-{time.time_ns()}")
    env = dict(os.environ, ALCYON_DATA_DIR=f"{HOME}/v62/build/data")
    alc = chess.engine.SimpleEngine.popen_uci(f"{HOME}/v62/build/alcyon", env=env, timeout=600)
    alc.configure({"Hash": 16, "Threads": 1})
    sf = chess.engine.SimpleEngine.popen_uci(f"{HOME}/sfraw/src/stockfish", timeout=600)
    sf.configure({"Threads": 1, "Hash": 16})
    # second processus Alcyon, uniquement pour l'eval statique (commande "eval")
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

    chunk = 0

    def new_chunk():
        nonlocal chunk
        chunk += 1
        tag = f"w{wid:03d}_{int(time.time())}_{chunk:04d}"
        return (tag, gzip.open(os.path.join(a.out, tag + ".plain.gz.tmp"), "wt"),
                gzip.open(os.path.join(a.out, tag + ".diag.gz.tmp"), "wt"),
                gzip.open(os.path.join(a.out, tag + ".leaf.gz.tmp"), "wt"))

    def close_chunk(tag, out, diag, leaf):
        out.close(); diag.close(); leaf.close()
        for ext in (".plain.gz", ".diag.gz", ".leaf.gz"):
            os.rename(os.path.join(a.out, tag + ext + ".tmp"), os.path.join(a.out, tag + ext))

    tag, out, diag, leaf = new_chunk()
    stat = open(os.path.join(a.out, f"w{wid:03d}.stat"), "w")
    n_pos = 0
    t0 = time.time()
    gi = 0
    while gi < a.games_per_worker:
        line = rnd.choice(openings).split()
        board = chess.Board(" ".join(line[:4]) + " 0 1")
        alc_color = rnd.choice([chess.WHITE, chess.BLACK])
        recs, drecs, lrecs = [], [], []
        pending = None  # dernier coup d'Alcyon en attente de l'avis de SF
        mate_res = None
        win_run = draw_run = 0  # adjudication (SF a faible profondeur ne convertit pas toujours)
        def over(b):
            return (b.is_game_over() or b.halfmove_clock >= 100 or b.is_repetition(3))
        while not over(board) and board.ply() < 400:
            if board.turn == alc_color:
                r = alc.play(board, chess.engine.Limit(depth=a.alc_depth),
                             info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
                s = r.info.get("score")
                a_sc = s.relative.score(mate_score=32000) if s else None
                take_leaf = rnd.random() * a.sample < 1.0
                pending = [board.fen(), r.move.uci(), a_sc]
                # feuille de la variante principale : la position dont l'eval statique a fixe le score
                pv = r.info.get("pv") or [r.move]
                lb = board.copy()
                n_pv = 0
                for m in pv:
                    if not lb.is_legal(m):
                        break
                    lb.push(m)
                    n_pv += 1
                if take_leaf and n_pv >= 1 and not lb.is_check() and not lb.is_game_over():
                    st_ev = static_eval(lb.fen())
                    li = sf.analyse(lb, chess.engine.Limit(depth=a.sf_depth))
                    ls = li["score"].relative
                    if st_ev is not None and not ls.is_mate():
                        lrecs.append((lb.fen(), li["pv"][0].uci(), ls.score(), lb.turn, n_pv, st_ev, a_sc))
                board.push(r.move)
            else:
                sampled = rnd.random() * a.sample < 1.0
                info = sf.analyse(board, chess.engine.Limit(depth=a.sf_depth if sampled else a.sf_fast_depth))
                sc = info["score"].relative
                mv = info["pv"][0]
                if pending is not None and sampled:
                    # avis de SF sur la position apres le coup d'Alcyon, cote Alcyon
                    drecs.append(pending + [-sc.score(mate_score=32000), board.ply()])
                    pending = None
                if sc.is_mate():
                    mate_res = "1-0" if (sc.mate() > 0) == (board.turn == chess.WHITE) else "0-1"
                    break
                v = sc.score()
                win_run = win_run + 1 if abs(v) >= a.adj_win else 0
                draw_run = draw_run + 1 if (abs(v) <= a.adj_draw and board.ply() >= 80) else 0
                if win_run >= 3:
                    mate_res = "1-0" if (v > 0) == (board.turn == chess.WHITE) else "0-1"
                    break
                if draw_run >= 10:
                    mate_res = "1/2-1/2"
                    break
                if sampled and not board.is_check():
                    recs.append((board.fen(), mv.uci(), sc.score(), board.ply(), board.turn))
                board.push(mv)
        res = mate_res or board.result()
        if res == "*" and (board.halfmove_clock >= 100 or board.is_repetition(3)):
            res = "1/2-1/2"
        if res == "*":
            res = "1/2-1/2"
            if recs and abs(recs[-1][2]) > 1000:
                res = "1-0" if (recs[-1][2] > 0) == (recs[-1][4] == chess.WHITE) else "0-1"
        wr = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[res]
        for fen, mv, score, ply, turn in recs:
            r_stm = wr if turn == chess.WHITE else -wr
            out.write(f"fen {fen}\nmove {mv}\nscore {score}\nply {ply}\nresult {r_stm}\ne\n")
        a_res = wr if alc_color == chess.WHITE else -wr
        for fen, mv, a_sc, sf_sc, ply in drecs:
            diag.write(f"{fen}\t{mv}\t{a_sc}\t{sf_sc}\t{ply}\t{a_res}\n")
        # feuilles : fen, coup SF, score SF (camp au trait de la feuille), longueur PV,
        # eval statique Alcyon (camp au trait de la feuille), score racine Alcyon (son camp),
        # resultat de la partie (camp au trait de la feuille)
        for fen, mv, score, turn, n_pv, st_ev, a_sc in lrecs:
            r_stm = wr if turn == chess.WHITE else -wr
            leaf.write(f"{fen}\t{mv}\t{score}\t{n_pv}\t{st_ev}\t{a_sc}\t{r_stm}\n")
        n_pos += len(recs)
        gi += 1
        if gi % a.games_per_chunk == 0:
            close_chunk(tag, out, diag, leaf)
            tag, out, diag, leaf = new_chunk()
        stat.seek(0)
        stat.write(f"{gi} {n_pos} {time.time() - t0:.0f}\n")
        stat.flush()
    close_chunk(tag, out, diag, leaf)
    alc.quit(); sf.quit(); ev.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--games-per-worker", type=int, default=100000)
    ap.add_argument("--games-per-chunk", type=int, default=50)
    ap.add_argument("--alc-depth", type=int, default=7)
    ap.add_argument("--sample", type=float, default=3.0)
    ap.add_argument("--sf-fast-depth", type=int, default=5)
    ap.add_argument("--sf-depth", type=int, default=9)
    ap.add_argument("--adj-win", type=int, default=1500)
    ap.add_argument("--adj-draw", type=int, default=15)
    ap.add_argument("--book", default=f"{HOME}/UHO_4060_v2.epd")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    openings = open(a.book).read().splitlines()
    ps = [Process(target=worker, args=(i, a, openings)) for i in range(a.workers)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("termine")
