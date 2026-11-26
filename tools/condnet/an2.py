"""Erreur par type de position pour N modeles (colonnes apres fen, score, res, ply).
Usage : an2.py <evals.tsv> nom1 nom2 ... (le 1er sert de reference pour les ecarts)."""
import sys, chess, numpy as np
from multiprocessing import Pool
NAMES = sys.argv[2:]; V = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}
def feats(l):
    t = l.rstrip("\n").split("\t"); fen, sc = t[0], int(t[1]); ev = [int(x) for x in t[4:]]
    B = chess.Board(fen); us = B.turn
    cnt = lambda pt, c: len(B.pieces(pt, c))
    npm = {c: sum(V[pt] * cnt(pt, c) for pt in (2, 3, 4, 5)) for c in (0, 1)}
    pw = {c: cnt(1, c) for c in (0, 1)}; tot = npm[0] + npm[1]
    imb = (npm[us] + pw[us]) - (npm[not us] + pw[not us])
    q = cnt(5, 0) != cnt(5, 1); r = cnt(4, 0) != cnt(4, 1); mi = (cnt(2, 0) + cnt(3, 0)) != (cnt(2, 1) + cnt(3, 1))
    pw_ = cnt(1, 0) != cnt(1, 1)
    asc = abs(sc)
    return sc, ev, dict(
        bloc="condition vraie (pieces differentes)" if (q or r or mi) else "condition fausse",
        deseq=("pieces differentes" if (q or r or mi) else "seulement pions differents" if pw_ else "materiel identique"),
        phase="ouverture/milieu (>=40)" if tot >= 40 else "milieu leger (20-39)" if tot >= 20 else "finale (<20)",
        materiel="materiel egal" if imb == 0 else "ecart 1-2" if abs(imb) <= 2 else "ecart >=3",
        eval="equilibre <50" if asc < 50 else "leger 50-150" if asc < 150 else "net 150-400" if asc < 400 else "decisif >=400")
if __name__ == "__main__":
    with Pool(60) as p: R = p.map(feats, open(sys.argv[1]), chunksize=2000)
    sc = np.array([r[0] for r in R], float); E = np.array([r[1] for r in R], float)
    sig = lambda x: 1 / (1 + np.exp(-x / 400)); T = sig(sc)
    ks = [1.0] * len(NAMES)  # meme echelle pour tous les reseaux
    P = np.stack([sig(ks[j] * E[:, j]) for j in range(len(NAMES))], 1); err = (P - T[:, None]) ** 2 * 1e4
    print("echelles", dict(zip(NAMES, [round(k, 2) for k in ks])))
    print(f"ensemble n={len(R)} erreur (x1e4) " + "  ".join(f"{n} {err[:, j].mean():.2f}" for j, n in enumerate(NAMES)) + "\n")
    for key in ["bloc", "deseq", "phase", "materiel", "eval"]:
        print("==", key); cats = {}
        for i, r in enumerate(R): cats.setdefault(r[2][key], []).append(i)
        for c, ix in sorted(cats.items(), key=lambda kv: -len(kv[1])):
            m = err[ix].mean(0)
            print(f"  {c:38s} {100 * len(ix) / len(R):5.1f}% " + " ".join(f"{n} {m[j]:6.2f}" for j, n in enumerate(NAMES))
                  + "".join(f"  {NAMES[j]} vs {NAMES[0]} {100 * (m[j] / m[0] - 1):+.1f}%" for j in range(1, len(NAMES))))
        print()
