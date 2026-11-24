"""Melange 30 % distillation / 70 % test80 mars sur le serveur, a partir de
train_distill.binpack (deja sans la validation). Equivalent a make_mix.py."""
import os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binpack_chunks import index, write_chunks
import math
SF, DIS, MAR, FRAC = sys.argv[1], sys.argv[2], sys.argv[3], 0.3
OUT = sys.argv[4] if len(sys.argv) > 4 else "train_mix"
# --resample : resultats de la distillation re-tires selon la probabilite du score SF
# (meme formule que calculate_sf_loss), pour que lambda 0.6 equivaille a lambda 1 sur
# ces positions en moyenne : p >= 0.5 -> gain avec proba 2p-1, sinon nulle (et symetrique).
RESAMPLE = "--resample" in sys.argv
rr = random.Random(7)
def pf(score):
    s = lambda x: 1 / (1 + math.exp(-x))
    return 0.5 * (1 + s((score - 270) / 380) - s((-score - 270) / 380))
def resample(rec):
    lines = rec.split("\n")
    sc = int(next(l for l in lines if l.startswith("score "))[6:])
    p = pf(sc)
    if p >= 0.5:
        r = 1 if rr.random() < 2 * p - 1 else 0
    else:
        r = -1 if rr.random() < 1 - 2 * p else 0
    return "\n".join(f"result {r}" if l.startswith("result ") else l for l in lines)
rnd = random.Random(2026)
conv = lambda s, d: subprocess.run([SF, "convert", s, d], check=True, stdout=subprocess.DEVNULL)
def records(p):
    rec = []
    for l in open(p):
        rec.append(l)
        if l == "e\n":
            yield "".join(rec); rec = []
conv(DIS, "tmp_d.plain")
dis = list(records("tmp_d.plain"))
if RESAMPLE:
    dis = [resample(r) for r in dis]
need = int(len(dis) * (1 - FRAC) / FRAC)
mch = index(MAR); rnd.shuffle(mch)
write_chunks("tmp_m.binpack", [(MAR, o, z) for o, z in mch[: need // 350_000 + 1]])
conv("tmp_m.binpack", "tmp_m.plain")
mar = []
for r in records("tmp_m.plain"):
    if len(mar) >= need: break
    mar.append(r)
allr = dis + mar; rnd.shuffle(allr)
open(OUT + ".plain", "w").writelines(allr)
conv(OUT + ".plain", OUT + ".binpack")
for p in ["tmp_d.plain", "tmp_m.binpack", "tmp_m.plain", OUT + ".plain"]: os.remove(p)
print(f"distillation {len(dis)}, mars {len(mar)}, total {len(allr)} ({100*len(dis)/len(allr):.1f} % distillation)")
