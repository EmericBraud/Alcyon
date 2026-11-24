"""Regroupe les morceaux de generation en archives .tar (une par serie et par heure) pour
l'envoi sur Drive : Google limite le nombre de fichiers par seconde, pas le volume.

gpu-archive/aws-gen/<serie>/<fichiers .gz>  ->  gpu-archive/packed/<serie>/<serie>_<heure>.tar
Une archive n'est reconstruite que si la liste de ses fichiers a change (l'heure en cours
grossit a chaque passage ; les heures passees sont figees). Les fichiers sont deja compresses,
le tar ne recompresse pas.
"""
import os, re, sys, tarfile, time
from collections import defaultdict

L = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(L, "aws-gen")
DST = os.path.join(L, "packed")
EPOCH = re.compile(r"_(\d{10})_")

n_built = 0
for serie in sorted(os.listdir(SRC)):
    d = os.path.join(SRC, serie)
    if not os.path.isdir(d):
        continue
    groups = defaultdict(list)
    for f in os.listdir(d):
        if not f.endswith(".gz"):
            continue
        m = EPOCH.search(f)
        key = time.strftime("%Y%m%d-%H", time.gmtime(int(m.group(1)))) if m else "tout"
        groups[key].append(f)
    od = os.path.join(DST, serie)
    os.makedirs(od, exist_ok=True)
    for key, files in groups.items():
        files.sort()
        arc = os.path.join(od, f"{serie}_{key}.tar")
        lst = arc + ".list"
        want = "\n".join(files)
        if os.path.exists(arc) and os.path.exists(lst) and open(lst).read() == want:
            continue
        tmp = arc + ".tmp"
        with tarfile.open(tmp, "w") as t:
            for f in files:
                t.add(os.path.join(d, f), arcname=f)
        os.replace(tmp, arc)
        open(lst, "w").write(want)
        n_built += 1
print(f"pack : {n_built} archive(s) (re)construite(s)")
