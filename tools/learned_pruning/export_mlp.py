"""Exporte un MLP de train_mlp.py (fichier .pt) en header C++.

    python3 export_mlp.py model.pt src/engine/search/learned_prune_mlp_weights.hpp
    python3 export_mlp.py check model.pt dump.bin

La normalisation (mean / std des 75 entrees de fit.design) est repliee dans
la connexion lineaire et dans la premiere couche de la tete ; le 1/127 de l0
dans la couche l0. learned_prune.hpp (mlp_logit) doit reproduire
train_mlp.Net.forward. `check` compare au logit enregistre par un dump fait
avec learned_prune_model=1.
"""
import sys

import numpy as np
import torch

import fit
import train_mlp


def load(path):
    ck = torch.load(path, weights_only=False)
    return {k: v.double().numpy() for k, v in ck["state"].items()}, ck["mean"].astype(np.float64), ck["std"].astype(np.float64), ck["hidden"]


def arr(name, a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return f"    constexpr float {name} = {a:.8e}f;\n"
    shape = "".join(f"[{n}]" for n in a.shape)

    def fmt(x):
        if x.ndim == 1:
            return "{" + ", ".join(f"{v:.8e}f" for v in x) + "}"
        return "{" + ",\n     ".join(fmt(r) for r in x) + "}"
    return f"    constexpr float {name}{shape} = {fmt(a)};\n"


def export(pt, out):
    s, mean, std, hidden = load(pt)
    n = len(mean)
    lin_w = s["lin.weight"][0] / std
    lin_b = s["lin.bias"][0] - np.dot(s["lin.weight"][0], mean / std)
    h1 = s["head.0.weight"].copy()
    h1_b = s["head.0.bias"] - h1[:, hidden:hidden + n] @ (mean / std)
    h1[:, hidden:hidden + n] /= std
    with open(out, "w") as f:
        f.write(f"// Genere par tools/learned_pruning/export_mlp.py depuis {pt.split('/')[-1]} -- ne pas editer.\n")
        f.write("#pragma once\n\nnamespace learned_prune_mlp\n{\n")
        f.write(f"    constexpr int kHidden = {hidden};\n    constexpr int kDesign = {n};\n")
        f.write(f"    constexpr int kDepths = {train_mlp.N_DEPTHS};\n    constexpr int kHead = {h1.shape[0]};\n")
        # Transposee [1024][hidden] : mlp_logit ne parcourt que les l0 non nuls
        # (~10 %) et ajoute leur ligne de poids, vectorisee.
        f.write(arr("kL0WT", (s["l0.weight"] / 127.0).T))
        f.write(arr("kL0B", s["l0.bias"]))
        f.write(arr("kLinW", lin_w))
        f.write(arr("kLinB", lin_b))
        f.write(arr("kH1W", h1))
        f.write(arr("kH1B", h1_b))
        f.write(arr("kH2W", s["head.2.weight"][0]))
        f.write(arr("kH2B", s["head.2.bias"][0]))
        f.write("}\n")
    print(f"ecrit {out} : hidden={hidden}, design={n}")


def check(pt, dump):
    ck = torch.load(pt, weights_only=False)
    r, l0, idx = train_mlp.load(dump)
    d, x, dep, _, _, _ = train_mlp.tensors(r, l0, idx, ck["mean"], ck["std"])
    net = train_mlp.Net(d.shape[1], True, ck["hidden"])
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        z = net(d, x, dep).numpy()
    err = np.abs(z - r["learned_z"][idx])
    print(f"{len(idx)} noeuds, ecart max {err.max():.2e}, moyen {err.mean():.2e}")
    assert err.max() < 1e-3, "le moteur et train_mlp.py ne calculent pas le meme MLP"


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    else:
        export(sys.argv[1], sys.argv[2])
