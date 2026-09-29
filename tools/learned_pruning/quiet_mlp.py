"""MLP l0 pour les coups calmes tardifs, un expert par tranche de profondeur.

    python3 quiet_mlp.py <dossier de dumps game_NNN.bin (+ .l0)> [every] [epochs]

Meme forme que le MoE retenu (train_safe.MoENet) : couche l0 (1024 -> 16)
partagee, puis par tranche de profondeur une tete (16 + scalaires -> 32 -> 1),
plus une partie lineaire sur les scalaires. l0 = position APRES le coup.
Classes equilibrees par tranche (pos_weight). Compare, sur les parties de
test, au rang (profondeur x rang) et au gradient boosting scalaire : part du
cout des coups calmes tardifs retirable a 0.1 / 0.5 / 1 / 2 % de coups
utiles perdus, classement par P(utile) / cout attendu (quiet_fit.py).
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier

import quiet_fit as qf

BANDS = qf.BANDS


def load(folder):
    tr, te = [], []
    for path in sorted(glob.glob(os.path.join(folder, "game_*.bin"))):
        r = np.fromfile(path, dtype=qf.DTYPE)
        l0 = np.fromfile(path + ".l0", dtype=np.uint8).reshape(-1, 1024)
        n = min(len(r), len(l0))  # une fin de fichier coupee ne doit pas decaler les deux
        (tr if int(os.path.basename(path)[5:8]) < 260 else te).append((r[:n], l0[:n]))
    cat = lambda parts: (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))  # noqa: E731
    return cat(tr), cat(te)


def band_of(r):
    b = np.zeros(len(r), np.int64)
    for i, (lo, hi) in enumerate(BANDS):
        b[(r["depth"] >= lo) & (r["depth"] <= hi)] = i
    return b


class Net(nn.Module):
    def __init__(self, n_x, hidden=16):
        super().__init__()
        self.l0 = nn.Linear(1024, hidden)
        self.lin = nn.Linear(n_x, len(BANDS))
        self.h1 = nn.ModuleList(nn.Linear(hidden + n_x, 32) for _ in BANDS)
        self.h2 = nn.ModuleList(nn.Linear(32, 1) for _ in BANDS)

    def forward(self, l0, x, b):
        h = torch.relu(self.l0(l0.float() / 127.0))
        z = self.lin(x).gather(1, b[:, None]).squeeze(1)
        hx = torch.cat([h, x], 1)
        out = torch.zeros_like(z)
        for i in range(len(BANDS)):
            m = b == i
            if m.any():
                out[m] = self.h2[i](torch.relu(self.h1[i](hx[m]))).squeeze(1)
        return z + out


def main():
    folder = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    (tr, l0tr), (te, l0te) = load(folder)
    wtr, wte = qf.weight(tr, every), qf.weight(te, every)
    ytr, yte = tr["useful"], te["useful"]
    btr, bte = band_of(tr), band_of(te)
    print(f"train {len(tr)} coups, test {len(te)} ; utiles (pondere) {np.average(yte, weights=wte):.2%}")

    xtr, xte = qf.features(tr), qf.features(te)
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    ntr, nte = (xtr - mean) / std, (xte - mean) / std
    # pos_weight par tranche : classes equilibrees dans chaque expert.
    pw = np.array([(1 - ytr[btr == i].mean()) / max(ytr[btr == i].mean(), 1e-6) for i in range(len(BANDS))])
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    L0tr, Xtr, Ytr, Btr = T(l0tr, torch.uint8), T(ntr), T(ytr.astype(np.float32)), T(btr, torch.int64)
    L0te, Xte, Bte = T(l0te, torch.uint8), T(nte), T(bte, torch.int64)
    PW = torch.from_numpy(pw.astype(np.float32))

    net = Net(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    bs = 16384

    def predict():
        with torch.no_grad():
            return torch.cat([torch.sigmoid(net(L0te[i:i + bs], Xte[i:i + bs], Bte[i:i + bs]))
                              for i in range(0, len(te), bs)]).numpy()

    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        tot = 0.0
        for i in range(0, len(tr), bs):
            j = perm[i:i + bs]
            z = net(L0tr[j], Xtr[j], Btr[j])
            loss = nn.functional.binary_cross_entropy_with_logits(z, Ytr[j], pos_weight=None,
                                                                  weight=1 + Ytr[j] * (PW[Btr[j]] - 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(j)
        p = predict()
        auc = [qf.curve(p[bte == i], yte[bte == i], te["subtree"][bte == i].astype(np.float64), wte[bte == i])[2]
               for i in range(len(BANDS))]
        print(f"  epoch {ep + 1}/{epochs} : loss {tot / len(tr):.4f} ; retirable a 1 % par tranche "
              + " ".join(f"{a:.1%}" for a in auc), flush=True)

    # References sur les memes donnees : rang et gradient boosting scalaire, par tranche.
    key = lambda r: np.minimum(r["depth"], 20) * 256 + np.minimum(r["rank"], 255)  # noqa: E731
    ktr, kte = key(tr), key(te)
    den = np.bincount(ktr, weights=wtr, minlength=21 * 256)
    rank_rate = (np.bincount(ktr, weights=wtr * ytr, minlength=21 * 256) + 1e-3) / (den + 1)
    cost_hat = (np.bincount(ktr, weights=wtr * tr["subtree"], minlength=21 * 256) + 1) / (den + 1)
    hgb = np.zeros(len(te))
    rng = np.random.default_rng(0)
    for i in range(len(BANDS)):
        m = np.nonzero(btr == i)[0]
        sub = rng.choice(m, min(len(m), 4_000_000), replace=False)
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, class_weight="balanced")
        g.fit(xtr[sub], ytr[sub])
        hgb[bte == i] = g.predict_proba(xte[bte == i])[:, 1]
    p = predict()
    cost = te["subtree"].astype(np.float64)
    ranks = [("rang / cout", rank_rate[kte] / cost_hat[kte]),
             ("GB scalaire / cout", hgb / cost_hat[kte]),
             ("MLP l0 / cout", p / cost_hat[kte])]
    head = "  ".join(f"perte {x:.1%}" for x in qf.LOSSES)
    for i, (lo, hi) in enumerate(BANDS):
        m = bte == i
        print(f"\n  depth {lo}-{hi} ({m.sum()} coups, utiles {np.average(yte[m], weights=wte[m]):.2%}, "
              f"{(cost[m] * wte[m]).sum() / (cost * wte).sum():.0%} du cout)\n    {'':20s}{head}")
        for name, s in ranks:
            print(f"    {name:20s}" + "  ".join(f"{v:10.1%}" for v in qf.curve(s[m], yte[m], cost[m], wte[m])))
    torch.save({"state": net.state_dict(), "mean": mean, "std": std, "bands": BANDS}, os.path.join(folder, "quiet_mlp.pt"))


if __name__ == "__main__":
    main()
