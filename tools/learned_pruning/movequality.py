"""Qualite du coup a budget egal : "le meme coup OU un aussi bon".

    python3 movequality.py prepare <dossier budget_eval> <config> [config...]
    python3 movequality.py report  <dossier budget_eval> <config> [config...]

prepare : pour chaque position, reunit les coups choisis par la base
(nb_base) et les configs (nb_<config>), joue chacun (python-chess) et ecrit
les positions resultantes dans <dossier>/mq/lot_XXXX (une par ligne) avec
leur cle dans <dossier>/mq/keys.tsv. Un moteur les cherche ensuite
(bench <profondeur>, mecanisme eteint) -> mq/eval_lot_XXXX.log.

report : score d'un coup = - score de la position apres le coup. Le meilleur
des coups evalues sert de reference ; perte = E(meilleur) - E(coup), avec
E(s) = 1 / (1 + 10^(-s/400)) : un coup different mais aussi bon ne coute
rien, une gaffe coute beaucoup. Donne aussi la profondeur atteinte a budget
egal (colonne depth des logs nodesfile) : ce que l'elagage achete.
"""
import glob
import os
import re
import sys

import numpy as np

LINE = re.compile(r"info string bench (\d+)/\d+ nodes (\d+) .* bestmove (\S+) score (-?\d+) depth (\d+)")
MATE, CLIP = 10000, 1500


def runs(out, name):
    """(lot, i) -> (coup, profondeur, noeuds)"""
    r = {}
    for path in glob.glob(os.path.join(out, f"{name}__lot_*.log")):
        lot = os.path.basename(path)[:-4].split("__")[1]
        for g in LINE.finditer(open(path).read()):
            r[(lot, int(g[1]))] = (g[3], int(g[5]), int(g[2]))
    return r


def fens(out):
    f = {}
    for path in glob.glob(os.path.join(out, "lots", "lot_*")):
        lot = os.path.basename(path)
        for i, line in enumerate(open(path), 1):
            f[(lot, i)] = line.strip()
    return f


def prepare(out, configs):
    import chess
    pos = fens(out)
    names = ["nb_base"] + ["nb_" + c for c in configs]
    all_runs = [runs(out, n) for n in names]
    os.makedirs(os.path.join(out, "mq"), exist_ok=True)
    keys = open(os.path.join(out, "mq", "keys.tsv"), "w")
    lines, terminal = [], []
    for k in sorted(pos):
        moves = {r[k][0] for r in all_runs if k in r and r[k][0] != "(none)"}
        for mv in sorted(moves):
            b = chess.Board(pos[k])
            b.push_uci(mv)
            if b.is_game_over(claim_draw=False):
                # Mat : le coup gagne ; pat / materiel insuffisant : nul.
                terminal.append((k, mv, MATE if b.is_checkmate() else 0))
                continue
            keys.write(f"{k[0]}\t{k[1]}\t{mv}\n")
            lines.append(b.fen())
    keys.close()
    for n in range(0, len(lines), 25):
        with open(os.path.join(out, "mq", f"lot_{n // 25:05d}"), "w") as f:
            f.write("\n".join(lines[n:n + 25]) + "\n")
    with open(os.path.join(out, "mq", "terminal.tsv"), "w") as f:
        for k, mv, s in terminal:
            f.write(f"{k[0]}\t{k[1]}\t{mv}\t{s}\n")
    print(f"{len(lines)} positions a evaluer, {len(terminal)} terminales")


def report(out, configs):
    keys = [l.rstrip("\n").split("\t") for l in open(os.path.join(out, "mq", "keys.tsv"))]
    evals = {}
    for path in glob.glob(os.path.join(out, "mq", "eval_lot_*.log")):
        n = int(os.path.basename(path)[9:-4])
        for g in LINE.finditer(open(path).read()):
            evals[n * 25 + int(g[1]) - 1] = int(g[4])
    score = {}  # (lot, i) -> {coup: score pour le camp qui joue le coup}
    for idx, (lot, i, mv) in enumerate(keys):
        if idx in evals:
            score.setdefault((lot, int(i)), {})[mv] = -evals[idx]
    for l in open(os.path.join(out, "mq", "terminal.tsv")):
        lot, i, mv, s = l.split("\t")
        score.setdefault((lot, int(i)), {})[mv] = int(s)

    def E(s):
        return 1 / (1 + 10 ** (-np.clip(s, -CLIP, CLIP) / 400))

    names = ["nb_base"] + ["nb_" + c for c in configs]
    all_runs = {n: runs(out, n) for n in names}
    common = sorted(set.intersection(*(set(r) for r in all_runs.values())) & set(score))
    common = [k for k in common if all(all_runs[n][k][0] in score[k] for n in names)]
    best = {k: E(max(score[k].values())) for k in common}

    def loss(n):
        return np.array([best[k] - E(score[k][all_runs[n][k][0]]) for k in common])

    base = loss("nb_base")
    base_depth = np.mean([all_runs["nb_base"][k][1] for k in common])
    print(f"{len(common)} positions ; perte en points de score espere (x100), plus bas = mieux\n")
    print(f"{'config':16s} {'depth':>6s} {'perte':>7s} {'base':>7s} {'ecart':>7s} {'IC 95 %':>16s}")
    print(f"{'base':16s} {base_depth:6.2f} {100 * base.mean():7.3f}")
    for c in configs:
        n = "nb_" + c
        l = loss(n)
        d = 100 * (l - base)
        se = d.std(ddof=1) / np.sqrt(len(d))
        depth = np.mean([all_runs[n][k][1] for k in common])
        print(f"{c:16s} {depth:6.2f} {100 * l.mean():7.3f} {100 * base.mean():7.3f} {d.mean():+7.3f} "
              f"[{d.mean() - 1.96 * se:+6.3f}, {d.mean() + 1.96 * se:+6.3f}]")


if __name__ == "__main__":
    {"prepare": prepare, "report": report}[sys.argv[1]](sys.argv[2], sys.argv[3:])
