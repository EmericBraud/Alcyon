"""Fenetre d'aspiration apprise, avec l'accumulateur de la racine.

Cible : log(1 + |score(d+1) - score(d)|). Entrees : signaux de recherche (comme
train_asp.py) + l0 de la racine (1024 octets, une fois par coup). Architecture :
l0 -> 16 (ReLU), calcule une seule fois par coup dans le moteur ; puis
[16 + scalaires] -> 32 -> 1 a chaque iteration. Compare avec / sans l0 a
architecture egale (le "sans" remplace l0 par des zeros).

Usage : python train_asp_l0.py DATA.tsv OUT_DIR    (DATA.tsv.l0 a cote)
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

SCALARS = ["depth", "stable", "frac_best", "frac_second", "abs_score", "abs_drop",
           "root_moves", "sel_minus_depth", "pieces"]


def load(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path)]
    a = np.array([r[:14] for r in rows if len(r) == 16], dtype=np.float64)
    g, ply, d, el, soft, nodes, st, fb, fs, sc, pv, rm, sel, pc = a.T
    k = g * 1000 + ply
    same = np.r_[k[1:] == k[:-1], False]
    nxt = np.r_[sc[1:], 0]
    m = same & (d >= 5) & (np.abs(sc) < 1500) & (np.abs(nxt) < 1500)
    scc, pvc = np.clip(sc, -2000, 2000), np.clip(pv, -2000, 2000)
    X = np.column_stack([d, np.minimum(st, 20), fb, fs, np.abs(scc), np.abs(pvc - scc), rm, sel - d, pc])[m]
    y = np.log1p(np.abs(nxt - sc))[m]

    raw = np.fromfile(path + ".l0", dtype=np.uint8)
    raw = raw[: len(raw) // 1032 * 1032].reshape(-1, 8 + 1024)
    ids = raw[:, :8].copy().view(np.int32).reshape(-1, 2)
    key_l0 = ids[:, 0].astype(np.int64) * 1000 + ids[:, 1]
    order = np.argsort(key_l0)
    pos = np.searchsorted(key_l0[order], k[m])
    pos = np.clip(pos, 0, len(order) - 1)
    found = key_l0[order][pos] == k[m]
    l0_idx = order[pos]
    return X[found], raw[:, 8:], l0_idx[found], y[found], g[m][found].astype(np.int64)


class Net(nn.Module):
    def __init__(self, n_scal, use_l0):
        super().__init__()
        self.use_l0 = use_l0
        self.l0 = nn.Linear(1024, 16)
        self.h = nn.Linear(16 + n_scal, 32)
        self.o = nn.Linear(32, 1)

    def forward(self, l0, s):
        z = torch.relu(self.l0(l0)) if self.use_l0 else torch.zeros(s.shape[0], 16)
        return self.o(torch.relu(self.h(torch.cat([z, s], 1)))).squeeze(1)


def fit(use_l0, Xs, L0, idx, y, tr, te, epochs=int(os.environ.get("ASP_EPOCHS", "12"))):
    torch.manual_seed(0)
    net = Net(Xs.shape[1], use_l0)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    L0t = torch.tensor(L0, dtype=torch.uint8)
    idxt = torch.tensor(idx)
    trn = np.where(tr)[0]
    for ep in range(epochs):
        np.random.default_rng(ep).shuffle(trn)
        for b in range(0, len(trn), 4096):
            j = torch.tensor(trn[b:b + 4096])
            l0 = L0t[idxt[j]].float() / 127.0
            loss = ((net(l0, Xt[j]) - yt[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    with torch.no_grad():
        tei = torch.tensor(np.where(te)[0])
        pred = np.concatenate([net(L0t[idxt[tei[b:b + 65536]]].float() / 127.0, Xt[tei[b:b + 65536]]).numpy()
                               for b in range(0, len(tei), 65536)])
    return net, pred


def main():
    data, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    torch.set_num_threads(min(32, os.cpu_count()))
    X, L0, idx, y, g = load(data)
    mean, std = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mean) / std
    te = g % 5 == 0
    tr = ~te
    rep = {"rows": int(len(y)), "var": float(np.var(y[te]))}
    dl = np.expm1(y[te])
    for use in (False, True):
        net, pred = fit(use, Xs, L0, idx, y, tr, te)
        r = {"mse": float(np.mean((pred - y[te]) ** 2))}
        for c in (2, 3, 4):
            w = np.maximum(8, c * np.exp(pred))
            r[f"c{c}"] = {"coverage": float((dl <= w).mean()), "mean_width": float(w.mean())}
        rep["with_l0" if use else "without_l0"] = r
        if use:
            sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
            np.savez(os.path.join(out_dir, "asp_l0.npz"), mean=mean, scale=std, **sd)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
