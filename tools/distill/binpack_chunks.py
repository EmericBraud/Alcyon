"""Outils sur les chunks binpack ("BINP" + taille u32 LE + donnees) : indexation,
extraction de chunks choisis au hasard, melange de chunks de plusieurs fichiers.
Un binpack est une suite de chunks independants : en concatener est un binpack valide."""
import random, struct, sys


def index(path):
    out = []
    with open(path, "rb") as f:
        off = 0
        while True:
            h = f.read(8)
            if len(h) < 8:
                break
            assert h[:4] == b"BINP", (path, off)
            n = struct.unpack("<I", h[4:])[0]
            out.append((off, 8 + n))
            off += 8 + n
            f.seek(off)
    return out


def write_chunks(dst, items):
    """items : liste de (chemin, offset, taille)."""
    handles = {}
    with open(dst, "wb") as o:
        for p, off, sz in items:
            f = handles.setdefault(p, open(p, "rb"))
            f.seek(off)
            o.write(f.read(sz))
    for f in handles.values():
        f.close()
