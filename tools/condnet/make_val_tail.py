"""Validation tiree de la meme source que l'entrainement, jamais utilisee : des chunks pris au
hasard dans la FIN de chaque mois BT4 (au-dela de la part lue par le filtre plafonne a 600 M
positions sous condition par mois, + 3 % de marge). Usage : make_val_tail.py <sortie> <chunks/mois>"""
import struct, random, sys, os
# part du fichier (en positions) lue par filt_typ_cap pour atteindre 600 M positions gardees
LU = {"03-mar": 3231493850 / 4002807615, "04-apr": 3233663394 / 5617793487,
      "05-may": 3232236553 / 4034359673, "06-jun": 3232444643 / 3606037815}
out, per = sys.argv[1], int(sys.argv[2])
rnd = random.Random(7); w = open(out, "wb"); tot = 0
for m, frac in LU.items():
    p = f"/workspace/bt4/test80-2024-{m}-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack"
    idx = []; off = 0
    with open(p, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            n = struct.unpack("<I", h[4:])[0]; idx.append((off, 8 + n)); off += 8 + n; f.seek(off)
        start = int(len(idx) * min(frac + 0.03, 0.99))
        pick = rnd.sample(range(start, len(idx)), per)
        for i in sorted(pick):
            o, n = idx[i]; f.seek(o); w.write(f.read(n)); tot += n
    print(m, "chunks", len(idx), "debut de la zone vierge", start, "pris", sorted(pick), flush=True)
w.close(); print("ecrit", out, tot, "octets")
