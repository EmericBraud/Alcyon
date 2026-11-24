"""Analyse des parties Alcyon vs SF8 (PGN fastchess, commentaires {eval/prof temps})."""
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
    cp = (100000 - float(v)) if mate else float(v) * 100
    if sign == "-":
        cp = -cp
    return cp, int(d), float(t)


def material(b):
    return sum(VAL.get(p.piece_type, 0) for p in b.piece_map().values())


def phase(mat):
    return "ouverture/milieu (>=50)" if mat >= 50 else ("milieu (30-49)" if mat >= 30 else ("finale (<30)" if mat >= 12 else "finale pauvre (<12)"))


games = []
with open(sys.argv[1]) as f:
    while (g := chess.pgn.read_game(f)):
        games.append(g)

res = Counter()
stats = defaultdict(list)
drops = []          # (game idx, ply, own eval before, after, depth, time, mat, sf eval)
loss_first = []     # first decisive moment in lost games
conv_fail = []      # had >= +300 (alcyon POV, agreed by SF) but did not win
by_color = Counter()
term = Counter()
time_alc, time_sf, depth_alc, depth_sf = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
calib = []          # (alcyon eval, sf eval) both from alcyon POV, same ply neighborhood
for gi, g in enumerate(games):
    a_white = g.headers["White"].startswith("alcyon")
    r = g.headers["Result"]
    score = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}[r]
    if not a_white:
        score = 1 - score
    res[score] += 1
    by_color[("blancs" if a_white else "noirs", score)] += 1
    term[g.headers.get("Termination", "?")] += 1
    b = g.board()
    a_evals, s_evals = [], []  # (ply, cp alcyon POV)
    for node in g.mainline():
        mover_white = b.turn == chess.WHITE
        mat = material(b)
        p = parse(node.comment)
        alc_moved = mover_white == a_white
        if p:
            cp, d, t = p
            pov = cp if alc_moved else -cp  # eval in alcyon POV
            ph = phase(mat)
            if alc_moved:
                a_evals.append((b.ply(), pov, d, t, mat))
                time_alc[ph].append(t); depth_alc[ph].append(d)
            else:
                s_evals.append((b.ply(), pov, d, t, mat))
                time_sf[ph].append(t); depth_sf[ph].append(d)
        b.push(node.move)
    # calibration: alcyon eval vs next sf eval
    sd = {pl: v for pl, v, *_ in s_evals}
    for pl, v, *_ in a_evals:
        if pl + 1 in sd and abs(v) < 1000 and abs(sd[pl + 1]) < 1000:
            calib.append((v, sd[pl + 1]))
    # surprises: own eval drop between consecutive own moves
    for (p0, v0, d0, t0, m0), (p1, v1, d1, t1, m1) in zip(a_evals, a_evals[1:]):
        if abs(v0) < 400 and v0 - v1 >= 150:
            drops.append((gi, p0, v0, v1, d0, t0, m0, score))
    # first moment SF (referee) sees alcyon <= -150 in lost games
    if score == 0:
        first = next(((pl, v, m) for pl, v, d, t, m in s_evals if v <= -150), None)
        a_first = next(((pl, v) for pl, v, *_ in a_evals if v <= -150), None)
        loss_first.append((gi, first, a_first, a_white, g.headers["FEN"]))
    if score < 1:
        best = max((v for pl, v, d, t, m in s_evals if pl > 0), default=-1e9)
        best_a = max((v for pl, v, *_ in a_evals), default=-1e9)
        if best >= 300 and best_a >= 300:
            pl_best = next(pl for pl, v, *_ in s_evals if v == best)
            conv_fail.append((gi, score, best, best_a, pl_best))

n = len(games)
print(f"{n} parties | score Alcyon : V {res[1]} N {res[0.5]} D {res[0]} ({100*(res[1]+res[0.5]/2)/n:.1f} %)")
print("par couleur :", {k: v for k, v in sorted(by_color.items())})
print("terminaisons :", dict(term))
print("\nTemps moyen / profondeur moyenne par phase (Alcyon | SF8) :")
for ph in ["ouverture/milieu (>=50)", "milieu (30-49)", "finale (<30)", "finale pauvre (<12)"]:
    if time_alc[ph]:
        print(f"  {ph:25s} temps {st.mean(time_alc[ph]):.2f}s | {st.mean(time_sf[ph]):.2f}s   prof {st.mean(depth_alc[ph]):.1f} | {st.mean(depth_sf[ph]):.1f}   (n={len(time_alc[ph])})")

# calibration: slope of sf vs alcyon
xs, ys = zip(*calib)
mx, my = st.mean(xs), st.mean(ys)
slope = sum((x - mx) * (y - my) for x, y in calib) / sum((x - mx) ** 2 for x in xs)
print(f"\nCalibration : eval SF8 ~= {slope:.2f} x eval Alcyon (n={len(calib)}) ; |Alcyon| moyen {st.mean(map(abs, xs)):.0f} cp, |SF8| moyen {st.mean(map(abs, ys)):.0f} cp")
dis = [(x, y) for x, y in calib if (x > 100 and y < -100) or (x < -100 and y > 100)]
print(f"  desaccords de signe nets (>100 cp de part et d'autre) : {len(dis)} ({100*len(dis)/len(calib):.1f} %)")

print(f"\nSurprises (eval Alcyon chute >= 150 cp entre deux de ses coups, depuis |eval|<400) : {len(drops)}")
pc = Counter(phase(d[6]) for d in drops)
print("  par phase :", dict(pc))
print("  par resultat de partie :", dict(Counter(d[7] for d in drops)))
print("  profondeur moyenne au coup precedant la chute :", f"{st.mean(d[4] for d in drops):.1f}", " temps moyen :", f"{st.mean(d[5] for d in drops):.2f}s")

print(f"\nDefaites : {len(loss_first)}")
lp = Counter()
for gi, first, a_first, aw, fen in loss_first:
    if first is None:
        lp["SF8 ne voit jamais -150 (bascule tardive)"] += 1
    else:
        lp[phase(first[2])] += 1
print("  phase ou SF8 voit Alcyon <= -150 pour la 1re fois :", dict(lp))
lag = [(first[0] - a_first[0]) for gi, first, a_first, aw, fen in loss_first if first and a_first]
print("  avance (en plies) de SF8 sur Alcyon pour voir -150 : moyenne", f"{st.mean(lag):.1f}", "| SF8 voit d'abord :", sum(1 for x in lag if x < 0), "/", len(lag))
early = [x for x in loss_first if x[1] and x[1][0] <= 2]
print("  defaites deja perdues selon SF8 des la sortie du livre (ply<=2) :", len(early))

print(f"\nNon-conversions (SF8 et Alcyon voyaient Alcyon >= +300, partie non gagnee) : {len(conv_fail)}")
for c in conv_fail[:15]:
    print("  partie", c[0] + 1, "score", c[1], "SF8 max", int(c[2]), "Alcyon max", int(c[3]), "ply", c[4])
