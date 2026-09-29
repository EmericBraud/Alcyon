"""Filtre bon marche avant la recherche reduite du MoE (Model = 2), hors ligne.

    python3 gate_eval.py <test.bin> <moe.pt> [mlp.pt]

En partie, le mode ombre (reduit lance, jamais cru) coute deja -23 Elo : la
recherche reduite coute plus qu'elle ne rapporte. On ne la lance que si un
predicteur sans recherche (etape 1 : regression = learned_z du dump fait en
Model = 0, ou MLP l0) est assez sur de l'issue du noeud, |z1| >= g. Pour
chaque g : part des sondes lancees, cout des sondes (noeuds du reduit), gain
des noeuds crus (sous-arbre - reduit), part des confiances du MoE gardees,
erreurs -- a depth >= MoeMinDepth, seuils MoeZ du moteur, poids 1 / p du
tirage du dump (EVERY = 256).
"""
import sys

import numpy as np
import torch

import moe_diag
import train_mlp
import train_safe

MOE_Z = np.array([1290, 757, 890, 1317, 1009, 933, 677, 803, 398, 334]) / 100.0  # config.hpp
MIN_DEPTH, R, EVERY = 7, 2, 256


def mlp_logit(path, r, l0, idx):
    ck = torch.load(path, weights_only=False)
    d, x, p, _, _, _ = train_mlp.tensors(r, l0, idx, ck["mean"], ck["std"])
    net = train_mlp.Net(d.shape[1], True, ck["hidden"])
    net.load_state_dict(ck["state"])
    with torch.no_grad():
        return torch.cat([net(d[i:i + 8192], x[i:i + 8192], p[i:i + 8192]) for i in range(0, len(d), 8192)]).numpy()


def main():
    test, moe = sys.argv[1], sys.argv[2]
    k = R - 1
    r, l0, idx = train_mlp.load(test)
    idx = idx[r["depth"][idx] >= MIN_DEPTH]
    rt = r[idx]
    p, y = moe_diag.predict(moe, r, l0, idx, k, None, None)
    z_moe = np.log(np.clip(p, 1e-12, 1) / np.clip(1 - p, 1e-12, 1))
    red_fh = rt["red_score"][:, k] >= rt["beta"]
    bucket, _ = train_safe.buckets(rt, red_fh, "ds")
    trust = z_moe >= MOE_Z[bucket]
    wrong = trust & (y < 0.5)
    w = 1.0 / np.minimum(1.0, 2.0 ** (rt["depth"] - 1) / EVERY)
    red = rt["red_nodes"][:, k].astype(np.float64)
    full = rt["subtree"].astype(np.float64)
    tot = (w * full).sum()
    gain = np.maximum(full - red, 0)
    print(f"{len(idx)} noeuds depth >= {MIN_DEPTH} ; MoE croit {np.average(trust, weights=w):.1%} des noeuds")
    preds = {"regression": np.abs(rt["learned_z"])}
    if len(sys.argv) > 3:
        preds["mlp"] = np.abs(mlp_logit(sys.argv[3], r, l0, idx))
    for name, a in preds.items():
        print(f"\n== filtre {name} : sondes lancees si |z1| >= g (en % du cout complet)")
        print("  garde   sondes   cout sondes   dont non crues   gain   confiances gardees   erreurs / crus")
        for q in [100, 80, 60, 40, 30, 20, 10]:
            g = np.percentile(a, 100 - q) if q < 100 else -1
            m = a >= g
            probe = (w * red * m).sum() / tot
            waste = (w * red * (m & ~trust)).sum() / tot
            gn = (w * gain * (m & trust)).sum() / tot
            kept = (w * (m & trust)).sum() / max((w * trust).sum(), 1e-9)
            err = (w * (m & wrong)).sum() / max((w * (m & trust)).sum(), 1e-9)
            print(f"  {q:4d}%  g={g:5.2f}   {probe:9.2%}   {waste:12.2%}   {gn:6.2%}   {kept:14.1%}   {err:10.3%}")


if __name__ == "__main__":
    main()
