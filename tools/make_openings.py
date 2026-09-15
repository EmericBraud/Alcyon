#!/usr/bin/env python3
"""Génère une suite d'ouvertures EPD variée pour la génération de données.

Pourquoi : les parties self-play partant toutes de la position initiale ne
produisent que des positions de trajectoire — exactement le défaut reproché au
dataset Leela (voir docs/gpu-async-eval/ordering-hints-plan.md). Partir de
milliers de positions distinctes élargit la distribution vers ce que la
recherche visite réellement.

Méthode : N plis aléatoires depuis la position initiale, puis on rejette les
positions déjà déséquilibrées matériellement — un aléatoire pur produit des
ouvertures où un camp a déjà donné une pièce, et les parties qui en sortent
n'apprennent rien d'utile sur l'ordonnancement.

Usage:  tools/make_openings.py --count 5000 --plies 8 --out data/openings.epd
"""

import argparse
import random

import chess

# Valeurs juste pour le filtre de déséquilibre, pas une éval.
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
    if material_gap(board) > max_gap:
        return None
    return board


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--max-gap", type=int, default=1,
                        help="écart matériel maximum toléré, en pions (défaut 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/openings.epd")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seen, kept, tries = set(), [], 0
    while len(kept) < args.count:
        tries += 1
        if tries > args.count * 200:
            raise SystemExit(f"seulement {len(kept)} ouvertures après {tries} essais -- desserrer --max-gap")
        board = random_opening(args.plies, args.max_gap, rng)
        if board is None:
            continue
        epd = board.epd()
        if epd in seen:
            continue
        seen.add(epd)
        kept.append(epd)

    with open(args.out, "w") as f:
        f.write("\n".join(kept) + "\n")
    print(f"{len(kept)} ouvertures uniques ({args.plies} plis, écart <= {args.max_gap}) "
          f"en {tries} essais -> {args.out}")


if __name__ == "__main__":
    main()
