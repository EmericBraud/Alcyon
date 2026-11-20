"""Temps moyen consomme par coup, par moteur, depuis des PGN fastchess
(commentaires "{score/profondeur 0.123s}"). Sert a detecter un biais de
gestion du temps : un moteur qui joue globalement plus vite ou plus lentement.

Usage : python time_usage.py PGN [PGN...]
"""

import re
import sys
from collections import defaultdict

import chess.pgn

TIME = re.compile(r"([0-9.]+)s")


def main():
    tot = defaultdict(lambda: [0.0, 0, 0.0, 0])  # somme, n, somme coups 1-30, n coups 1-30
    for path in sys.argv[1:]:
        with open(path) as f:
            while True:
                g = chess.pgn.read_game(f)
                if g is None:
                    break
                names = {True: g.headers["White"], False: g.headers["Black"]}
                node, ply = g, 0
                while node.variations:
                    node = node.variations[0]
                    m = TIME.search(node.comment or "")
                    if m:
                        side = names[ply % 2 == 0]
                        t = float(m.group(1))
                        e = tot[side]
                        e[0] += t
                        e[1] += 1
                        if ply < 60:
                            e[2] += t
                            e[3] += 1
                    ply += 1
    for name, (s, n, s30, n30) in sorted(tot.items()):
        print(f"{name:10} {s / n:.3f} s/coup ({n} coups)   coups 1-30 : {s30 / max(1, n30):.3f} s")


if __name__ == "__main__":
    main()
