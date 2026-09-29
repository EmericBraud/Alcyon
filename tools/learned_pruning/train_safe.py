"""Label "reduire est sur" (docs/learned-pruning-bilan.md), dump v3.

    python3 train_safe.py <train.bin> <test.bin> <R> [--hidden H] [--epochs E] [--out safe_R.pt]

Mecanisme vise : chercher le noeud a depth - R, puis decider si l'on fait
confiance a ce resultat reduit (on s'arrete) ou non (recherche complete).
Le modele voit les features habituelles, l0, ET ce que la recherche reduite
a rendu (cote de beta, ecart a beta, cout). Label : le resultat reduit est
du meme cote de beta que le resultat complet.

Comparaison sur le test, a economie egale -- part des noeuds ou l'on saute
la recherche complete, et part du cout complet economise -- du taux
d'erreurs non detectees (on fait confiance, et le reduit avait tort) :
  toujours    : on fait toujours confiance au reduit
  pfh         : regle actuelle -- confiance si le reduit va dans le sens du
                modele P(fail-high) deploye (logit du dump), seuil sur |z|
  safe        : ce modele, seuil sur P(meme resultat)
"""
import os
import sys

import numpy as np
import torch
from torch import nn

import fit
import train_mlp

BATCH = 8192


def red_features(r, k):
    """Ce que la recherche reduite a rendu : cote, ecart a beta, cout."""
    red_fh = r["red_score"][:, k] >= r["beta"]
    margin = np.clip(r["red_score"][:, k] - r["beta"], -1500, 1500) / 100.0
    cost = np.log1p(r["red_nodes"][:, k]) / 10.0
    return np.stack([np.where(red_fh, 1.0, -1.0), margin, cost], 1).astype(np.float32), red_fh


def tensors(r, l0, idx, k, mean=None, std=None):
    rr = r[idx]
    m, x = fit.features(rr)
    extra, red_fh = red_features(rr, k)
    d = np.concatenate([fit.design(m, x, "b").astype(np.float32), extra], 1)
    if mean is None:
        mean, std = d.mean(0), d.std(0) + 1e-6
    d = (d - mean) / std
    depth = np.minimum(rr["depth"], train_mlp.N_DEPTHS) - 1
    full_fh = rr["outcome"] != 1
    y = (red_fh == full_fh).astype(np.float32)
    return (torch.from_numpy(d), torch.from_numpy(np.ascontiguousarray(l0[idx])),
            torch.from_numpy(depth.astype(np.int64)), torch.from_numpy(y), mean, std, red_fh, full_fh)


def curve(conf, wrong, full_cost, trust_mask=None):
    """Confiance decroissante : pour chaque prefixe, part des noeuds sautes,
    part du cout complet economise, part des noeuds en erreur non detectee."""
    order = np.argsort(-conf, kind="stable")
    if trust_mask is not None:
        order = order[trust_mask[order]]
    n = len(conf)
    skipped = np.arange(1, len(order) + 1) / n
    saved = np.cumsum(full_cost[order]) / full_cost.sum()
    errors = np.cumsum(wrong[order]) / n
    return skipped, saved, errors


def at_saved(curve_, targets):
    _, saved, errors = curve_
    out = []
    for t in targets:
        i = np.searchsorted(saved, t)
        out.append(errors[i] if i < len(errors) else float("nan"))
    return out


def main():
    train_path, test_path, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
    a = dict(zip(sys.argv[4::2], sys.argv[5::2]))
    hidden, epochs = int(a.get("--hidden", 16)), int(a.get("--epochs", 20))
    out = a.get("--out", f"safe_{R}.pt")
    k = R - 1
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    rtr, l0tr, itr = train_mlp.load(train_path)
    rte, l0te, ite = train_mlp.load(test_path)
    itr = train_mlp.subsample(rtr, itr, rng)
    dtr, xtr, ptr, ytr, mean, std, _, _ = tensors(rtr, l0tr, itr, k)
    dte, xte, pte, yte, _, _, red_fh, full_fh = tensors(rte, l0te, ite, k, mean, std)
    print(f"R={R} train {len(ytr)} test {len(yte)} ; le reduit a raison sur {yte.mean().item():.1%} du test", flush=True)

    net = train_mlp.Net(dtr.shape[1], hidden > 0, max(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps = epochs * ((len(ytr) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    loss_fn = nn.BCEWithLogitsLoss()
    ev = torch.from_numpy(np.random.default_rng(1).choice(len(yte), min(len(yte), 1_000_000), replace=False))
    for ep in range(epochs):
        perm = torch.randperm(len(ytr))
        total = 0.0
        for i in range(0, len(ytr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(net(dtr[b], xtr[b], ptr[b]), ytr[b])
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        with torch.no_grad():
            te_loss = loss_fn(net(dte[ev], xte[ev], pte[ev]), yte[ev]).item()
        print(f"  epoch {ep + 1:2d}/{epochs} : train {total / len(ytr):.4f}  test {te_loss:.4f}", flush=True)
    torch.save({"state": net.state_dict(), "mean": mean, "std": std, "hidden": hidden, "R": R}, out)

    p_safe = train_mlp.predict(net, dte, xte, pte)
    rt = rte[ite]
    wrong = red_fh != full_fh
    # Cout de la recherche complete sautee : sous-arbre du noeud moins les
    # recherches reduites du label.
    full_cost = np.maximum(rt["subtree"] - rt["red_nodes"].sum(1), 1).astype(np.float64)
    z = rt["learned_z"]
    pfh_agree = (z >= 0) == red_fh
    curves = {
        "toujours": curve(np.zeros(len(wrong)), wrong, full_cost),
        "pfh": curve(np.abs(z), wrong, full_cost, pfh_agree),
        "safe": curve(p_safe, wrong, full_cost),
    }
    targets = (0.10, 0.20, 0.30, 0.40, 0.50)
    print(f"\nerreurs non detectees (% des noeuds) a cout complet economise de {', '.join(f'{t:.0%}' for t in targets)}")
    for name, c in curves.items():
        errs = at_saved(c, targets)
        print(f"  {name:9s} " + "  ".join(f"{e:6.2%}" for e in errs))
    print("\npar profondeur, a 30 % du cout economise :")
    dep = rt["depth"]
    for lo, hi in fit.SLICES:
        s = (dep >= lo) & (dep <= hi)
        if s.sum() < 1000:
            continue
        row = []
        for name, conf, mask in (("pfh", np.abs(z), pfh_agree), ("safe", p_safe, None)):
            c = curve(conf[s], wrong[s], full_cost[s], None if mask is None else mask[s])
            row.append(f"{name} {at_saved(c, (0.30,))[0]:6.2%}")
        print(f"  depth {lo if lo == hi else f'{lo}+':>3} : " + "   ".join(row))


if __name__ == "__main__":
    main()
