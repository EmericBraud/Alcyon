"""Juge : evals de plusieurs moteurs (commande `eval` d'Alcyon) sur des positions d'arbres de
recherche notees par Leela BT4, puis erreur par type de position.
Usage : judge_eval.py <positions_bt4.tsv> <sortie.tsv> nom=lanceur [nom=lanceur ...]"""
import sys, subprocess, chess, numpy as np
from multiprocessing import Pool
src, out = sys.argv[1], sys.argv[2]
specs = [a.split("=", 1) for a in sys.argv[3:]]
rows = []
for l in open(src):
    f = l.rstrip("\n").split("\t"); sc = int(f[1])
    if sc == 32002 or abs(sc) >= 10000: continue
    rows.append((f[0], sc))
def run(args):
    cmd, batch = args
    inp = "".join(f"position fen {r[0]}\neval\n" for r in batch) + "quit\n"
    o = subprocess.run([cmd], input=inp, capture_output=True, text=True).stdout
    v = [int(x.split()[1]) for x in o.splitlines() if x.startswith("score:")]
    assert len(v) == len(batch), (cmd, len(v), len(batch)); return v
bs = [rows[i:i + 5000] for i in range(0, len(rows), 5000)]
E = {}
with Pool(120) as p:
    for n, c in specs:
        E[n] = [x for b in p.map(run, [(c, b) for b in bs]) for x in b]; print(n, "fait", flush=True)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        f.write(f"{r[0]}\t{r[1]}\t" + "\t".join(str(E[n][i]) for n, _ in specs) + "\n")
names = [n for n, _ in specs]
def feats(fen):
    B = chess.Board(fen); c = lambda p, col: len(B.pieces(p, col))
    typ = c(5, 1) != c(5, 0) or c(4, 1) != c(4, 0) or c(2, 1) + c(3, 1) != c(2, 0) + c(3, 0)
    n = len(B.piece_map())
    return typ, ("25-32" if n >= 25 else "17-24" if n >= 17 else "9-16" if n >= 9 else "<=8")
with Pool(120) as p: F = p.map(feats, [r[0] for r in rows], chunksize=5000)
sc = np.array([r[1] for r in rows], float); M = np.array([E[n] for n in names], float).T
sig = lambda x: 1 / (1 + np.exp(-x / 400)); T = sig(sc)
err = (sig(M) - T[:, None]) ** 2 * 1e4
ks = [min(np.arange(0.5, 2.01, 0.02), key=lambda k: np.mean((sig(k * M[:, j]) - T) ** 2)) for j in range(len(names))]
errk = np.stack([(sig(ks[j] * M[:, j]) - T) ** 2 * 1e4 for j in range(len(names))], 1)
print(f"\npositions {len(rows)}")
print("erreur (x1e4), echelle 1      : " + "  ".join(f"{n} {err[:, j].mean():.2f}" for j, n in enumerate(names)))
print("erreur (x1e4), echelle ajustee : " + "  ".join(f"{n} {errk[:, j].mean():.2f} (k {ks[j]:.2f})" for j, n in enumerate(names)))
for key, idx in [("pieces differentes", 0), ("nombre de pieces", 1)]:
    cats = {}
    for i, fe in enumerate(F): cats.setdefault(fe[idx], []).append(i)
    print("==", key)
    for c_, ix in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        m = err[ix].mean(0)
        print(f"  {str(c_):8s} {100 * len(ix) / len(rows):5.1f}% " + " ".join(f"{n} {m[j]:7.2f}" for j, n in enumerate(names)))
asc = np.abs(sc)
for lab, sel in [("|BT4| < 150", asc < 150), ("150-400", (asc >= 150) & (asc < 400)), (">= 400", asc >= 400)]:
    m = err[sel].mean(0); print(f"  {lab:12s} {100 * sel.mean():5.1f}% " + " ".join(f"{n} {m[j]:7.2f}" for j, n in enumerate(names)))
