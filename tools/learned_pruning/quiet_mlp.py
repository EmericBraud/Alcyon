"""MLP l0 pour les coups calmes tardifs, un expert par tranche de profondeur.

    python3 quiet_mlp.py <dossier de dumps game_NNN.bin (+ .l0)> [every] [epochs]

Meme forme que le MoE retenu (train_safe.MoENet) : couche l0 (1024 -> 16)
partagee, puis 12 experts (profondeur 1 / 2 / 3 / 4-6 / 7-9 / 10+ x noeud
cut / all), chacun une tete (16 + scalaires -> 32 -> 1), plus une partie
lineaire sur les scalaires. l0 = position APRES le coup. Entrainement pondere
par le poids de tirage, classes equilibrees par expert ; epoque choisie sur
la validation (parties 220-259), decroissance cosinus. Compare, sur les parties de
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

BANDS = qf.BANDS  # tranches du rapport final
DEPTHS = [(1, 1), (2, 2), (3, 3), (4, 6), (7, 9), (10, 64)]
N_EXPERTS = 2 * len(DEPTHS)  # x type de noeud (cut / all)


def load(folder):
    """Parties 0-219 entrainement, 220-259 validation (choix de l'epoque), 260+ test."""
    sets = ([], [], [])
    for path in sorted(glob.glob(os.path.join(folder, "game_*.bin"))):
        r = np.fromfile(path, dtype=qf.DTYPE)
        l0 = np.fromfile(path + ".l0", dtype=np.uint8).reshape(-1, 1024)
        n = min(len(r), len(l0))  # une fin de fichier coupee ne doit pas decaler les deux
        g = int(os.path.basename(path)[5:8])
        sets[0 if g < 220 else 1 if g < 260 else 2].append((r[:n], l0[:n]))
    cat = lambda parts: (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))  # noqa: E731
    return tuple(cat(x) for x in sets)


def expert_of(r):
    e = np.zeros(len(r), np.int64)
    for i, (lo, hi) in enumerate(DEPTHS):
        e[(r["depth"] >= lo) & (r["depth"] <= hi)] = i
    return 2 * e + (r["cut_node"] != 0)


def band_of(r):
    b = np.zeros(len(r), np.int64)
    for i, (lo, hi) in enumerate(BANDS):
        b[(r["depth"] >= lo) & (r["depth"] <= hi)] = i
    return b


class Net(nn.Module):
    def __init__(self, n_x, hidden=16):
        super().__init__()
        self.l0 = nn.Linear(1024, hidden)
        self.lin = nn.Linear(n_x, N_EXPERTS)
        self.h1 = nn.ModuleList(nn.Linear(hidden + n_x, 32) for _ in range(N_EXPERTS))
        self.h2 = nn.ModuleList(nn.Linear(32, 1) for _ in range(N_EXPERTS))

    def forward(self, l0, x, b):
        h = torch.relu(self.l0(l0.float() / 127.0))
        z = self.lin(x).gather(1, b[:, None]).squeeze(1)
        hx = torch.cat([h, x], 1)
        out = torch.zeros_like(z)
        for i in range(N_EXPERTS):
            m = b == i
            if m.any():
                out[m] = self.h2[i](torch.relu(self.h1[i](hx[m]))).squeeze(1)
        return z + out


def main():
    folder = sys.argv[1]
    every = sys.argv[2] if len(sys.argv) > 2 else "4096"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    (tr, l0tr), (va, l0va), (te, l0te) = load(folder)
    wtr, wva, wte = qf.weight(tr, every), qf.weight(va, every), qf.weight(te, every)
    ytr, yva, yte = tr["useful"], va["useful"], te["useful"]
    etr, eva, ete = expert_of(tr), expert_of(va), expert_of(te)
    btr, bte = band_of(tr), band_of(te)
    print(f"train {len(tr)}, validation {len(va)}, test {len(te)} coups ; utiles (pondere) {np.average(yte, weights=wte):.2%}")
    for i in range(N_EXPERTS):
        m = etr == i
        lo, hi = DEPTHS[i // 2]
        print(f"  expert depth {lo}-{hi} {'cut' if i % 2 else 'all'} : {m.sum()} coups, utiles {np.average(ytr[m], weights=wtr[m]):.2%}")

    xtr, xva, xte = qf.features(tr), qf.features(va), qf.features(te)
    mean, std = xtr.mean(0), xtr.std(0) + 1e-6
    # Poids d'entrainement : poids de tirage (meme distribution que l'evaluation),
    # normalise par expert, puis classes equilibrees dans chaque expert.
    sw = np.zeros(len(tr))
    for i in range(N_EXPERTS):
        m = etr == i
        if not m.any():
            continue
        w = wtr[m] / wtr[m].mean()
        pos = np.average(ytr[m], weights=w)
        sw[m] = w * np.where(ytr[m] == 1, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))
    T = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    L0tr, Xtr, Ytr, Etr, Wtr = T(l0tr, torch.uint8), T((xtr - mean) / std), T(ytr.astype(np.float32)), T(etr, torch.int64), T(sw)
    evalset = lambda l0, x, e: (T(l0, torch.uint8), T((x - mean) / std), T(e, torch.int64))  # noqa: E731
    VA, TE = evalset(l0va, xva, eva), evalset(l0te, xte, ete)

    net = Net(xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 16384

    def predict(ds):
        with torch.no_grad():
            return torch.cat([torch.sigmoid(net(ds[0][i:i + bs], ds[1][i:i + bs], ds[2][i:i + bs]))
                              for i in range(0, len(ds[0]), bs)]).numpy()

    def score(p, r, y, w, lvl):
        c = r["subtree"].astype(np.float64)
        b = band_of(r)
        return [qf.curve(p[b == i], y[b == i], c[b == i], w[b == i])[lvl] for i in range(len(BANDS))]

    best, best_state = -1.0, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(tr))
        tot = wsum = 0.0
        for i in range(0, len(tr), bs):
            j = perm[i:i + bs]
            loss = (nn.functional.binary_cross_entropy_with_logits(net(L0tr[j], Xtr[j], Etr[j]), Ytr[j], reduction="none") * Wtr[j]).sum()
            opt.zero_grad()
            (loss / Wtr[j].sum()).backward()
            opt.step()
            tot += loss.item()
            wsum += Wtr[j].sum().item()
        sched.step()
        pv = predict(VA)
        vloss = -np.average(yva * np.log(np.clip(pv, 1e-7, 1)) + (1 - yva) * np.log(np.clip(1 - pv, 1e-7, 1)), weights=wva)
        s01, s1 = score(pv, va, yva, wva, 0), score(pv, va, yva, wva, 2)
        # Critere de l'epoque : part retirable a 0.1 % et 1 %, moyennee sur les tranches.
        crit = (np.mean(s01) + np.mean(s1)) / 2
        if crit > best:
            best, best_state = crit, {k: v.clone() for k, v in net.state_dict().items()}
        print(f"  epoch {ep + 1}/{epochs} : train {tot / wsum:.4f}  val logloss {vloss:.4f}  "
              f"val retirable 0.1 % {' '.join(f'{a:.1%}' for a in s01)}  1 % {' '.join(f'{a:.1%}' for a in s1)}"
              f"{'  *' if crit == best else ''}", flush=True)
    net.load_state_dict(best_state)

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
        sub = rng.choice(m, min(len(m), 4_000_000), replace=False)  # train seul (0-219)
        g = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, class_weight="balanced")
        g.fit(xtr[sub], ytr[sub])
        hgb[bte == i] = g.predict_proba(xte[bte == i])[:, 1]
    p = predict(TE)
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
