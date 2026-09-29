"""Exporte un MoE de train_safe.py (--moe dsp) en header C++.

    python3 export_moe.py model.pt src/engine/search/learned_prune_moe_weights.hpp
    python3 export_moe.py check model.pt dump.bin

Entrees (78) : fit.design(..., "b") puis les 3 features du resultat reduit
(train_safe.red_features, R du modele). Normalisation repliee dans la partie
lineaire et dans la couche h1 ; 1/127 replie dans la couche l0 (stockee
transposee, parcourue seulement sur les l0 non nuls). learned_prune.hpp
(moe_logit) doit reproduire train_safe.MoENet.forward. `check` compare au
logit enregistre par un dump v3 fait avec learned_prune_model=2 et le label.
"""
import sys

import numpy as np
import torch

import export_mlp
import train_mlp
import train_safe


def export(pt, out):
    ck = torch.load(pt, weights_only=False)
    assert ck["moe"] == "dsp", "seul le schema dsp est gere par le moteur"
    s = {k: v.double().numpy() for k, v in ck["state"].items()}
    mean, std, hidden = ck["mean"].astype(np.float64), ck["std"].astype(np.float64), ck["hidden"]
    n = len(mean)
    lin_w = s["lin.weight"][0] / std
    lin_b = s["lin.bias"][0] - np.dot(s["lin.weight"][0], mean / std)
    h1 = s["h1.weight"].copy()
    h1_b = s["h1.bias"] - h1[:, hidden:hidden + n] @ (mean / std)
    h1[:, hidden:hidden + n] /= std
    n_b = s["h2_w"].shape[0]
    arr = export_mlp.arr
    with open(out, "w") as f:
        f.write(f"// Genere par tools/learned_pruning/export_moe.py depuis {pt.split('/')[-1]} -- ne pas editer.\n")
        f.write("#pragma once\n\nnamespace learned_prune_moe\n{\n")
        f.write(f"    constexpr int kR = {ck['R']};\n    constexpr int kHidden = {hidden};\n    constexpr int kDesign = {n};\n")
        f.write(f"    constexpr int kDepths = {train_mlp.N_DEPTHS};\n    constexpr int kBuckets = {n_b};\n    constexpr int kHead = 32;\n")
        f.write(arr("kL0WT", (s["l0.weight"] / 127.0).T))
        f.write(arr("kL0B", s["l0.bias"]))
        f.write(arr("kLinW", lin_w))
        f.write(arr("kLinB", lin_b))
        f.write(arr("kH1W", h1))
        f.write(arr("kH1B", h1_b))
        f.write(arr("kH2W", s["h2_w"]))
        f.write(arr("kH2B", s["h2_b"]))
        f.write("}\n")
    print(f"ecrit {out} : R={ck['R']}, {n_b} experts, hidden={hidden}, design={n}")


def check(pt, dump):
    ck = torch.load(pt, weights_only=False)
    k = ck["R"] - 1
    r, l0, idx = train_mlp.load(dump)
    d, x, p, _, _, _, _, _, b = train_safe.tensors(r, l0, idx, k, ck["mean"], ck["std"], ck["moe"])
    net = train_safe.MoENet(d.shape[1], ck["hidden"], ck["state"]["h2_w"].shape[0])
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        z = net(d, x, p, b).numpy()
    err = np.abs(z - r["learned_z"][idx])
    print(f"{len(idx)} noeuds, ecart max {err.max():.2e}, moyen {err.mean():.2e}")
    assert err.max() < 1e-3, "le moteur et train_safe.py ne calculent pas le meme MoE"


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
    else:
        export(sys.argv[1], sys.argv[2])
