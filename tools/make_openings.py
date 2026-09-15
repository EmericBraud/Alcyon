#!/usr/bin/env python3
"""Génère une suite d'ouvertures EPD variée pour la génération de données.

Pourquoi : des parties self-play partant toutes de la position initiale ne
produisent que des positions de trajectoire — exactement le défaut reproché au
dataset Leela (voir docs/gpu-async-eval/ordering-hints-plan.md). Partir de
milliers de positions distinctes élargit la distribution.

Méthode : N plis aléatoires, puis DEUX filtres.

Le filtre matériel seul ne suffit pas, et c'est mesuré : avec lui seul, 2
ouvertures sur 10 donnaient des parties démarrant à |éval| > 3, dont une à
+8.65. Une dame en l'air laisse l'écart matériel à zéro. Or une partie qui
commence décidée n'apprend rien sur l'ordonnancement — le camp gagnant gagne
quel que soit l'ordre des coups.

D'où le second filtre, par évaluation : chaque candidate est soumise au moteur
à faible profondeur et rejetée si |score| dépasse le seuil. Coût : une
recherche par candidate (profondeur 6 par défaut, quelques ms), un seul
process moteur, séquentiel.

Usage:
  tools/make_openings.py --count 5000 --out data/openings.epd \\
      --engine build_metal/chess26
"""

import argparse
import random
import subprocess

import chess

# Valeurs pour le pré-filtre matériel uniquement, ce n'est pas une éval.
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_gap(board):
    gap = 0
    for piece_type, value in PIECE_VALUE.items():
        gap += value * (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK)))
    return abs(gap)


def random_opening(plies, max_gap, rng):
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            return None
        board.push(rng.choice(moves))
        if board.is_game_over():
            return None
    return board if material_gap(board) <= max_gap else None


class EngineFilter:
    """Un seul process moteur, interrogé séquentiellement.

    `go` est non bloquant côté moteur, donc on lit jusqu'à `bestmove` et on
    retient le dernier `score cp` vu. OwnBook est coupé : un coup de livre
    revient sans ligne info, donc sans score.
    """

    def __init__(self, path, depth):
        self.depth = depth
        self.proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name OwnBook value false")
        self._send("setoption name Threads value 1")

    def _send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until(self, token):
        lines = []
        for line in self.proc.stdout:
            lines.append(line)
            if line.startswith(token) or token in line:
                return lines
        raise RuntimeError(f"le moteur a ferme son flux en attendant '{token}'")

    def score_cp(self, epd):
        self._send(f"position fen {epd} 0 1")
        self._send(f"go depth {self.depth}")
        score = None
        for line in self._read_until("bestmove"):
            if " score cp " in line:
                score = int(line.split(" score cp ")[1].split()[0])
            elif " score mate " in line:
                score = 100000
        return score

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=1,
                        help="pré-filtre : écart matériel max en pions (défaut 1)")
    parser.add_argument("--engine", default=None,
                        help="binaire moteur pour le filtre par éval ; sans lui, "
                             "seul le pré-filtre matériel s'applique (insuffisant, voir docstring)")
    parser.add_argument("--filter-depth", type=int, default=6)
    parser.add_argument("--max-eval", type=int, default=120,
                        help="|score| max en centipions pour accepter une ouverture (défaut 120)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/openings.epd")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    engine = EngineFilter(args.engine, args.filter_depth) if args.engine else None
    if engine is None:
        print("ATTENTION : pas de --engine, filtre par eval desactive -- "
              "mesure: ~20 % des ouvertures sortent alors a |eval| > 3")

    seen, kept, tries, rejected_eval = set(), [], 0, 0
    try:
        while len(kept) < args.count:
            tries += 1
            if tries > args.count * 500:
                raise SystemExit(f"seulement {len(kept)} ouvertures apres {tries} essais -- "
                                 f"desserrer --max-eval ou --max-gap")
            board = random_opening(args.plies, args.max_gap, rng)
            if board is None:
                continue
            epd = board.epd()
            if epd in seen:
                continue
            seen.add(epd)
            if engine is not None:
                score = engine.score_cp(epd)
                if score is None or abs(score) > args.max_eval:
                    rejected_eval += 1
                    continue
            kept.append(epd)
            if len(kept) % 500 == 0:
                print(f"  {len(kept)}/{args.count}...", flush=True)
    finally:
        if engine is not None:
            engine.close()

    with open(args.out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"{len(kept)} ouvertures uniques ({args.plies} plis, ecart <= {args.max_gap}"
          + (f", |eval| <= {args.max_eval}cp a profondeur {args.filter_depth}" if engine else "")
          + f") en {tries} essais, {rejected_eval} rejetees par l'eval -> {args.out}")


if __name__ == "__main__":
    main()
