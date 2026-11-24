"""Dataset final de fine-tuning : distillation (Alcyon contre SF) melangee a test80 (v6).
- Validation : quelques chunks de distillation mis de cote (val_distill) + chunks de fevrier
  (val_feb, la meme source que la perte de reference de v5).
- Entrainement : le reste de la distillation + des chunks de mars tires au hasard, en quantite
  telle que la distillation fasse --frac du total ; melange au niveau des POSITIONS (et non des
  chunks : un chunk test80 = ~400 k positions d'affilee), puis reconversion en binpack.
Usage : python3 make_mix.py --sf PATH_SF_TOOLS --frac 0.3
"""
import argparse, glob, os, random, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from binpack_chunks import index, write_chunks

ap = argparse.ArgumentParser()
ap.add_argument("--sf", required=True)
ap.add_argument("--distill", default="../aws-ds/final")
ap.add_argument("--mar", default="mar.binpack")
ap.add_argument("--feb", default="feb.binpack")
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--val-distill-chunks", type=int, default=10)
ap.add_argument("--val-feb-chunks", type=int, default=3)
a = ap.parse_args()
rnd = random.Random(2026)


def conv(src, dst):
    subprocess.run([a.sf, "convert", src, dst], check=True, stdout=subprocess.DEVNULL)


def count_plain(p):
    return sum(1 for l in open(p) if l == "e\n")


# distillation : chunks, validation mise de cote
dch = [(f, o, z) for f in sorted(glob.glob(f"{a.distill}/part*.binpack")) for o, z in index(f)]
rnd.shuffle(dch)
write_chunks("val_distill.binpack", dch[: a.val_distill_chunks])
write_chunks("tmp_distill.binpack", dch[a.val_distill_chunks:])
fch = index(a.feb)
write_chunks("val_feb.binpack", [(a.feb, o, z) for o, z in rnd.sample(fch, a.val_feb_chunks)])

conv("tmp_distill.binpack", "tmp_distill.plain")
n_d = count_plain("tmp_distill.plain")
need = int(n_d * (1 - a.frac) / a.frac)
mch = index(a.mar)
rnd.shuffle(mch)
# ~400 k positions par chunk de mars : on prend un chunk de plus puis on tronque a 'need'
k = max(1, need // 350_000 + 1)
write_chunks("tmp_mar.binpack", [(a.mar, o, z) for o, z in mch[:k]])
conv("tmp_mar.binpack", "tmp_mar.plain")


def records(p):
    rec = []
    for l in open(p):
        rec.append(l)
        if l == "e\n":
            yield "".join(rec)
            rec = []


mar = []
for r in records("tmp_mar.plain"):
    if len(mar) >= need:
        break
    mar.append(r)
dis = list(records("tmp_distill.plain"))
allr = dis + mar
rnd.shuffle(allr)
with open("train_mix.plain", "w") as f:
    f.writelines(allr)
conv("train_mix.plain", "train_mix.binpack")
print(f"distillation {len(dis)} ({100 * len(dis) / len(allr):.1f} %), mars {len(mar)}, total {len(allr)}")
for p in ["tmp_distill.binpack", "tmp_distill.plain", "tmp_mar.binpack", "tmp_mar.plain", "train_mix.plain"]:
    os.remove(p)
for p in ["val_distill.binpack", "val_feb.binpack"]:
    conv(p, "tmp.plain"); print(p, count_plain("tmp.plain")); os.remove("tmp.plain")
