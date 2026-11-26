"""Analyse du juge sur positions calmes : aucune piece en prise (attaquee par moins cher ou non
defendue) pour aucun des deux camps, et |score BT4| < 1500. Usage : judge_an.py <evals.tsv> noms..."""
import sys, chess, numpy as np
from multiprocessing import Pool
names = sys.argv[2:]; V = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
def f(l):
    t = l.rstrip("\n").split("\t"); B = chess.Board(t[0]); sc = int(t[1]); ev = [int(x) for x in t[2:]]
    hang = 0
    for s, p in B.piece_map().items():
        if p.piece_type == 6: continue
        att = B.attackers(not p.color, s)
        if att and (not B.attackers(p.color, s) or min(V[B.piece_at(x).piece_type] or 99 for x in att) < V[p.piece_type]): hang += 1
    c = lambda pt, col: len(B.pieces(pt, col))
    typ = c(5, 1) != c(5, 0) or c(4, 1) != c(4, 0) or c(2, 1) + c(3, 1) != c(2, 0) + c(3, 0)
    return sc, ev, hang, typ, len(B.piece_map())
with Pool(120) as p: R = p.map(f, open(sys.argv[1]), chunksize=5000)
sc = np.array([r[0] for r in R], float); E = np.array([r[1] for r in R], float)
hang = np.array([r[2] for r in R]); typ = np.array([r[3] for r in R]); npc = np.array([r[4] for r in R])
sig = lambda x: 1 / (1 + np.exp(-x / 400)); T = sig(sc); err = (sig(E) - T[:, None]) ** 2 * 1e4
def show(lab, sel):
    m = err[sel].mean(0)
    print(f"  {lab:40s} {sel.sum():9d} " + "  ".join(f"{n} {m[j]:7.2f}" for j, n in enumerate(names)))
print("positions", len(R))
show("toutes", np.ones(len(R), bool))
quiet = (hang == 0) & (np.abs(sc) < 1500)
show("calmes (0 piece en prise, |BT4|<1500)", quiet)
show("calmes, pieces identiques", quiet & ~typ)
show("calmes, pieces differentes", quiet & typ)
show("calmes, 17-32 pieces", quiet & (npc >= 17))
show("calmes, <= 16 pieces", quiet & (npc <= 16))
show("1 piece en prise, |BT4|<1500", (hang == 1) & (np.abs(sc) < 1500))
show("2+ pieces en prise, |BT4|<1500", (hang >= 2) & (np.abs(sc) < 1500))
