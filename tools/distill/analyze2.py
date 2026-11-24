"""Erreurs d'Alcyon vues par SF8 (arbitre) : saut de l'eval SF8 entre deux coups SF8
encadrant un coup d'Alcyon. Seuils en unites SF8 (Alcyon ~3.5x plus grand)."""
import re, sys, statistics as st
from collections import Counter, defaultdict
import chess, chess.pgn

CMT = re.compile(r"([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+) ([\d.]+)s")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def parse(c):
    m = CMT.search(c or "")
    if not m:
        return None
    sign, mate, v, d, t = m.groups()
    cp = (3000 if mate else min(float(v) * 100, 3000))
    return (-cp if sign == "-" else cp), int(d), float(t)


def mat(b):
    return sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values())


def ph(m):
    return "A ouverture (>=50)" if m >= 50 else ("B milieu (30-49)" if m >= 30 else ("C finale (12-29)" if m >= 12 else "D finale pauvre"))


games = []
with open(sys.argv[1]) as f:
    while (g := chess.pgn.read_game(f)):
        games.append(g)

errs = []  # (gi, phase, sf_before, sf_after, a_eval, a_depth, a_time, score, ply, san, capture?, check?, piece)
forfeit = []
pairs = defaultdict(list)
clock_left = []
for gi, g in enumerate(games):
    aw = g.headers["White"].startswith("alcyon")
    s = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}[g.headers["Result"]]
    s = s if aw else 1 - s
    pairs[g.headers["FEN"]].append(s)
    if g.headers.get("Termination") == "time forfeit":
        forfeit.append((gi + 1, "Alcyon" if s == 0 else "SF8", g.headers["PlyCount"]))
    b = g.board()
    last_sf = None
    pending = None
    for node in g.mainline():
        a_moves = (b.turn == chess.WHITE) == aw
        p = parse(node.comment)
        m = mat(b)
        if p:
            cp, d, t = p
            pov = cp if a_moves else -cp
            if a_moves:
                pending = (last_sf, pov, d, t, m, b.ply(), b.san(node.move), b.is_capture(node.move), b.gives_check(node.move))
            else:
                if pending and pending[0] is not None:
                    before = pending[0]
                    after = pov
                    if abs(before) < 250 and before - after >= 100:
                        errs.append((gi, ph(pending[4]), before, after, pending[1], pending[2], pending[3], s, pending[5], pending[6], pending[7], pending[8]))
                last_sf = pov
                pending = None
        b.push(node.move)

print("pertes au temps :", forfeit)
pc = Counter()
for v in pairs.values():
    if len(v) == 2:
        pc[sum(v)] += 1
print("paires (meme ouverture, deux couleurs) somme du score Alcyon :", dict(sorted(pc.items())))

print(f"\nErreurs d'Alcyon selon SF8 (eval SF8 baisse >= 100 cp SF apres un coup d'Alcyon, depuis |eval| < 250) : {len(errs)} sur {len(games)} parties")
print("  par phase :", dict(sorted(Counter(e[1] for e in errs).items())))
print("  par resultat final :", dict(Counter(e[7] for e in errs)))
big = [e for e in errs if e[2] - e[3] >= 200]
print(f"  dont grosses (>= 200 cp SF) : {len(big)}, par phase :", dict(sorted(Counter(e[1] for e in big).items())))
print(f"  profondeur Alcyon sur ces coups : {st.mean(e[5] for e in errs):.1f} (moyenne generale ailleurs ~17)  temps : {st.mean(e[6] for e in errs):.2f}s")
print(f"  eval Alcyon au moment du coup (ses unites) : moyenne {st.mean(e[4] for e in errs):.0f} ; Alcyon se croyait >= 0 : {sum(1 for e in errs if e[4] >= 0)}/{len(errs)}")
print("  coups de capture :", sum(e[10] for e in errs), " echecs :", sum(e[11] for e in errs))
print("\nExemples (partie, ply, coup, SF8 avant->apres, Alcyon eval/prof/temps) :")
for e in sorted(errs, key=lambda e: e[3] - e[2])[:25]:
    print(f"  p{e[0]+1:3d} ply {e[8]:3d} {e[9]:7s} {e[1]:18s} SF8 {e[2]:+5.0f} -> {e[3]:+5.0f}   Alcyon {e[4]:+6.0f} d{e[5]} {e[6]:.2f}s  score {e[7]}")
