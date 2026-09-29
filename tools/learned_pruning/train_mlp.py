"""Etape MLP de docs/learned-pruning.md : l0 (accumulateur NNUE) + scalaires.

    python3 train_mlp.py <train.bin> <test.bin> [epochs]
    python3 train_mlp.py <train.bin> <test.bin> --hidden H --epochs E --out pred_H.npy

La seconde forme (entrainement long) : un seul modele (H = 0 : lin seul),
taux d'apprentissage en cosinus, loss de test a chaque epoch, predictions
sur tout le test ecrites dans --out pour compare_mlp.py.

Les .bin viennent de run_dump.sh ; chacun a son .bin.l0 (1024 octets par
enregistrement, meme ordre). Deux modeles, meme pipeline, meme evaluation :

  lin : logistique sur fit.design(..., "b") -- la regression deja integree
  mlp : lin + un reseau sur [l0 -> 32, scalaires, profondeur] -> 32 -> 1

lin est un cas particulier de mlp (la tete peut s'annuler) : tout ecart
vient de l0. Critere : celui de fit.py, par profondeur.
"""
import os
import sys

import numpy as np
import torch
from torch import nn

import fit

MAX_PER_DEPTH = 1_000_000  # ponytail: plafond par profondeur, garde l'entrainement en minutes
BATCH = 16384
N_DEPTHS = len(fit.SLICES)


def load(path):
    """Enregistrements filtres comme fit.load, et leurs l0 alignes."""
    r = np.fromfile(path, dtype=fit.dtype_for(path))
    l0 = np.memmap(path + ".l0", dtype=np.uint8, mode="r", shape=(len(r), 1024))
    ok = (np.abs(r["beta"]) < 9000) & (np.abs(r["static_eval"]) < 9000)
    return r, l0, np.nonzero(ok)[0]


def subsample(r, idx, rng):
    keep = []
    for lo, hi in fit.SLICES:
        s = idx[(r["depth"][idx] >= lo) & (r["depth"][idx] <= hi)]
        keep.append(s if len(s) <= MAX_PER_DEPTH else rng.choice(s, MAX_PER_DEPTH, replace=False))
    return np.sort(np.concatenate(keep))


def tensors(r, l0, idx, mean=None, std=None):
    m, x = fit.features(r[idx])
    d = fit.design(m, x, "b").astype(np.float32)
    if mean is None:
        mean, std = d.mean(0), d.std(0) + 1e-6
    d = (d - mean) / std
    depth = np.minimum(r["depth"][idx], N_DEPTHS) - 1
    y = (r["outcome"][idx] != 1).astype(np.float32)  # NMP compte comme fail-high
    return (torch.from_numpy(d), torch.from_numpy(np.ascontiguousarray(l0[idx])),
            torch.from_numpy(depth.astype(np.int64)), torch.from_numpy(y), mean, std)


class Net(nn.Module):
    def __init__(self, n_design, use_l0, hidden=32):
        super().__init__()
        self.lin = nn.Linear(n_design, 1)
        self.use_l0 = use_l0
        if use_l0:
            self.l0 = nn.Linear(1024, hidden)
            self.head = nn.Sequential(nn.Linear(hidden + n_design + N_DEPTHS, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, d, l0, depth):
        z = self.lin(d).squeeze(1)
        if self.use_l0:
            h = torch.relu(self.l0(l0.float() / 127.0))
            onehot = nn.functional.one_hot(depth, N_DEPTHS).float()
            z = z + self.head(torch.cat([h, d, onehot], 1)).squeeze(1)
        return z


def train(net, d, l0, depth, y, epochs):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(d[b], l0[b], depth[b]), y[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        print(f"    epoch {ep + 1}/{epochs} : loss {total / n:.4f}", flush=True)


@torch.no_grad()
def predict(net, d, l0, depth):
    return torch.cat([torch.sigmoid(net(d[i:i + BATCH], l0[i:i + BATCH], depth[i:i + BATCH]))
                      for i in range(0, len(d), BATCH)]).numpy()


def main(train_path, test_path, epochs=4):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rtr, l0tr, itr = load(train_path)
    rte, l0te, ite = load(test_path)
    itr = subsample(rtr, itr, rng)
    print(f"train {len(itr)} noeuds, test {len(ite)} noeuds", flush=True)
    dtr, xtr, ptr, ytr, mean, std = tensors(rtr, l0tr, itr)
    dte, xte, pte, yte, _, _ = tensors(rte, l0te, ite, mean, std)
    preds = {}
    for name, use_l0 in (("lin", False), ("mlp", True)):
        print(f"  {name}", flush=True)
        net = Net(dtr.shape[1], use_l0)
        train(net, dtr, xtr, ptr, ytr, epochs)
        preds[name] = predict(net, dte, xte, pte)
        torch.save({"state": net.state_dict(), "mean": mean, "std": std}, f"{name}.pt")
    y, sub, dep = yte.numpy() > 0.5, rte["subtree"][ite], rte["depth"][ite]
    print(f"\n{'depth':>6s} {'modele':>6s} {'logloss':>8s}  part du sous-arbre coupable a precision >= 95 / 98 / 99 %")
    for lo, hi in fit.SLICES:
        s = (dep >= lo) & (dep <= hi)
        if not s.any():
            continue
        name_d = f"{lo}" if lo == hi else f"{lo}+"
        for name, p in preds.items():
            ps = np.clip(p[s], 1e-7, 1 - 1e-7)
            ll = -np.mean(y[s] * np.log(ps) + (~y[s]) * np.log(1 - ps))
            cov = " / ".join(f"{fit.coverage(p[s], y[s], sub[s], t)[1]:5.1%}" for t in (0.95, 0.98, 0.99))
            print(f"{name_d:>6s} {name:>6s} {ll:8.4f}  {cov}")


def long_run(train_path, test_path, hidden, epochs, out, lr=2e-3, batch=8192):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rtr, l0tr, itr = load(train_path)
    rte, l0te, ite = load(test_path)
    itr = subsample(rtr, itr, rng)
    dtr, xtr, ptr, ytr, mean, std = tensors(rtr, l0tr, itr)
    dte, xte, pte, yte, _, _ = tensors(rte, l0te, ite, mean, std)
    # Loss de test suivie sur un sous-ensemble fixe (1M), pour rester rapide.
    ev = torch.from_numpy(np.random.default_rng(1).choice(len(yte), min(len(yte), 1_000_000), replace=False))
    net = Net(dtr.shape[1], hidden > 0, max(hidden, 1))
    n_params = sum(p.numel() for p in net.parameters())
    print(f"hidden={hidden} params={n_params} train={len(ytr)} test={len(yte)}", flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    steps = epochs * ((len(ytr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        perm = torch.randperm(len(ytr))
        total = 0.0
        net.train()
        for i in range(0, len(ytr), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(dtr[b], xtr[b], ptr[b]), ytr[b])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        net.eval()
        with torch.no_grad():
            p = predict(net, dte[ev], xte[ev], pte[ev])
            te_loss = loss_fn(torch.logit(torch.from_numpy(np.clip(p, 1e-7, 1 - 1e-7))), yte[ev]).item()
        print(f"  epoch {ep + 1:2d}/{epochs} : train {total / len(ytr):.4f}  test {te_loss:.4f}", flush=True)
    np.save(out, predict(net, dte, xte, pte))
    torch.save({"state": net.state_dict(), "mean": mean, "std": std, "hidden": hidden}, out.replace(".npy", ".pt"))


if __name__ == "__main__":
    if "--hidden" in sys.argv:
        a = dict(zip(sys.argv[3::2], sys.argv[4::2]))
        long_run(sys.argv[1], sys.argv[2], int(a["--hidden"]), int(a["--epochs"]), a["--out"])
    else:
        main(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
