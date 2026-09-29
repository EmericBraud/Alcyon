"""Biais de contexte des dumps : mode bench vs mode partie, memes positions.

    python3 context_bias.py prep   <games.pgn> <dossier> <n_parties> [ply_min]
    python3 context_bias.py play   <binaire> <partie.txt> <profondeur>
    python3 context_bias.py report <bench.bin> <partie.bin>

Tous nos dumps viennent du mode bench : chaque position cherchee de zero
(TT vide) avec une racine en fenetre complete. En partie, la TT arrive pleine
des coups precedents et la racine passe par l'aspiration. Les features TT
etant les plus fortes, ce biais peut fausser le modele.

prep : pour chaque partie, <dossier>/game_NNN.txt (un coup UCI par ligne, a
partir du debut) et positions.txt (les FEN des positions jouees, ply >=
ply_min, dans l'ordre -- pour le mode bench).
play : rejoue la partie coup par coup ("position startpos moves ..." puis
"go depth D", sans ucinewgame : TT conservee), a partir de ply_min. Le dump
se regle par l'environnement (ALCYON_PRUNE_DUMP...), comme pour le bench.
report : compare les distributions au point de decision, par profondeur, et
le taux d'erreur du resultat reduit (label v3).
"""
import os
import subprocess
import sys

import numpy as np

PLY_MIN = 8


def prep(pgn, out, n_games, ply_min=PLY_MIN):
    import chess.pgn
    os.makedirs(out, exist_ok=True)
    fens = []
    with open(pgn) as f:
        for g in range(int(n_games)):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            b = game.board()
            moves = []
            for mv in game.mainline_moves():
                if len(moves) >= int(ply_min):
                    fens.append(b.fen())
                moves.append(mv.uci())
                b.push(mv)
            with open(os.path.join(out, f"game_{g:03d}.txt"), "w") as gf:
                gf.write(f"{ply_min}\n" + "\n".join(moves) + "\n")
    with open(os.path.join(out, "positions.txt"), "w") as pf:
        pf.write("\n".join(fens) + "\n")
    print(f"{g + 1} parties, {len(fens)} positions")


def play(engine, game_file, depth):
    lines = open(game_file).read().split()
    ply_min, moves = int(lines[0]), lines[1:]
    p = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    p.stdin.write("uci\nsetoption name Threads value 1\nsetoption name OwnBook value false\n"
                  "setoption name learned_prune_enabled value 0\nisready\n")
    while p.stdout.readline().strip() != "readyok":
        pass
    p.stdin.write("ucinewgame\n")  # une fois, au debut de la partie
    for ply in range(ply_min, len(moves)):
        p.stdin.write(f"position startpos moves {' '.join(moves[:ply])}\ngo depth {depth}\n")
        while True:
            line = p.stdout.readline()
            if not line:  # le moteur est mort : ne pas boucler indefiniment
                sys.exit(f"{game_file} : moteur mort au ply {ply} (code {p.wait()}), "
                         f"apres les coups {' '.join(moves[max(0, ply - 3):ply])}")
            if line.startswith("bestmove"):
                break
    p.stdin.write("quit\n")
    p.wait()


def report(bench_path, game_path):
    import fit
    b, g = fit.load(bench_path), fit.load(game_path)
    print(f"bench : {len(b)} noeuds ; partie : {len(g)} noeuds\n")

    def stats(r):
        found = r["tt_found"] != 0
        margin = np.clip(r["static_eval"] - r["beta"], -1500, 1500)
        out = {
            "TT trouvee": found.mean(),
            "TT depth >= depth": (found & (r["tt_depth"] >= r["depth"])).mean(),
            "TT depth - depth (si trouvee)": np.mean(r["tt_depth"][found] - r["depth"][found]) if found.any() else np.nan,
            "borne TT exacte": (found & (r["tt_flag"] == 0)).mean(),
            "borne TT haute (alpha)": (found & (r["tt_flag"] == 1)).mean(),
            "borne TT basse (beta)": (found & (r["tt_flag"] == 2)).mean(),
            "|eval - beta| median": np.median(np.abs(margin)),
            "noeud cut": r["cut_node"].mean(),
            "eval ply-2 connue": (r["eval_prev2"] != fit.K_EVAL_NONE).mean(),
            "fail-high (issue)": (r["outcome"] != 1).mean(),
        }
        if "red_score" in r.dtype.names:
            fh = r["outcome"] != 1
            for k, R in ((0, 1), (1, 2)):
                out[f"erreur du reduit R={R}"] = ((r["red_score"][:, k] >= r["beta"]) != fh).mean()
        return out

    for name, (lo, hi) in (("toutes", (1, 99)), ("depth 1-3", (1, 3)), ("depth 4-6", (4, 6)), ("depth 7+", (7, 99))):
        sb = b[(b["depth"] >= lo) & (b["depth"] <= hi)]
        sg = g[(g["depth"] >= lo) & (g["depth"] <= hi)]
        if len(sb) < 100 or len(sg) < 100:
            continue
        xb, xg = stats(sb), stats(sg)
        print(f"== {name} (bench {len(sb)}, partie {len(sg)})")
        for k in xb:
            print(f"  {k:32s} {xb[k]:8.3f}   {xg[k]:8.3f}")
        print()


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep(*sys.argv[2:])
    elif cmd == "play":
        play(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        report(sys.argv[2], sys.argv[3])
