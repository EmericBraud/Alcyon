"""Exporte la decision "coups calmes tardifs" en header C++ (learned_quiet_weights.hpp).

    python3 quiet_export.py <dossier dump equilibre> <probas de tirage> <sortie.hpp>
    python3 quiet_export.py check <quiet_mlp.pt> <dump.bin> <probas>   # parite (dump + .l0 + .z)

Contenu : tables par (profondeur <= 20, rang <= 255) de log(taux d'utilite)
et log(cout attendu), apprises sur les parties d'entrainement ; le MLP de
quiet_mlp.py (12 experts) avec sa recalibration (log des cotes a priori de
chaque expert) ; les seuils de log(P(utile) / cout attendu) qui, sur les
parties de test, font perdre 0.1 % et 1 % des coups utiles, pour chaque mode :
1 table, 2 MLP a depth <= 6 seulement, 3 MLP a depth <= 6 et table au-dela.
"""
import sys

import numpy as np
import torch

import quiet_fit as qf
import quiet_mlp as qm

MAX_D, MAX_R = 20, 255
LOSS_LEVELS = [0.001, 0.01]


def tables(tr, w):
    key = np.minimum(tr["depth"], MAX_D) * (MAX_R + 1) + np.minimum(tr["rank"], MAX_R)
    n = (MAX_D + 1) * (MAX_R + 1)
    den = np.bincount(key, weights=w, minlength=n)
    rate = (np.bincount(key, weights=w * tr["useful"], minlength=n) + 1e-3) / (den + 1)
    cost = (np.bincount(key, weights=w * tr["subtree"], minlength=n) + 1) / (den + 1)
    return np.log(rate), np.log(cost), key


def mlp_logit(ck, r, l0):
    net = qm.Net(len(ck["mean"]))
    net.load_state_dict(ck["state"])
    x = torch.from_numpy(((qf.features(r) - ck["mean"]) / ck["std"]).astype(np.float32))
    L0, E = torch.from_numpy(np.ascontiguousarray(l0)), torch.from_numpy(qm.expert_of(r))
    with torch.no_grad():
        z = torch.cat([net(L0[i:i + 16384], x[i:i + 16384], E[i:i + 16384]) for i in range(0, len(r), 16384)]).numpy()
    return z + ck["prior_logit"][qm.expert_of(r)]


def threshold(log_s, eligible, useful, w, loss):
    """Plus grand seuil tel que retirer les coups eligibles de log_s < seuil
    perde au plus `loss` des coups utiles (en poids, sur tous les coups)."""
    idx = np.nonzero(eligible)[0]
    order = idx[np.argsort(log_s[idx], kind="stable")]
    lost = np.cumsum((useful * w)[order]) / (useful * w).sum()
    k = np.searchsorted(lost, loss, side="right")
    return float(log_s[order[k - 1]]) if k > 0 else -1e9


def lit(v):
    t = f"{v:.9g}"
    return (t if any(c in t for c in ".en") else t + ".0") + "f"  # "-9f" n'est pas un litteral C++


def arr(name, a):
    a = np.asarray(a, dtype=np.float64)
    body = ", ".join(lit(v) for v in a.ravel())
    dims = "".join(f"[{d}]" for d in a.shape)
    return f"    constexpr float {name}{dims} = {{{body}}};\n"


def export(folder, probs, out):
    (tr, _), _, (te, l0te) = qm.load(folder)
    wtr, wte = qf.weight(tr, probs), qf.weight(te, probs)
    log_rate, log_cost, _ = tables(tr, wtr)
    ck = torch.load(f"{folder}/quiet_mlp.pt", weights_only=False)
    etr = qm.expert_of(tr)
    prior = np.array([np.average(tr["useful"][etr == i], weights=wtr[etr == i]) for i in range(qm.N_EXPERTS)])
    ck["prior_logit"] = np.log(prior / (1 - prior))
    torch.save(ck, f"{folder}/quiet_mlp.pt")

    kte = np.minimum(te["depth"], MAX_D) * (MAX_R + 1) + np.minimum(te["rank"], MAX_R)
    z = mlp_logit(ck, te, l0te)
    log_p_mlp = -np.logaddexp(0, -z)  # log sigmoid
    shallow = te["depth"] <= 6
    s_table = log_rate[kte] - log_cost[kte]
    s_mlp = log_p_mlp - log_cost[kte]
    s_hyb = np.where(shallow, s_mlp, s_table)
    all_ = np.ones(len(te), bool)
    y = te["useful"]
    thr = [[threshold(s_table, all_, y, wte, x) for x in LOSS_LEVELS],
           [threshold(s_mlp, shallow, y, wte, x) for x in LOSS_LEVELS],
           [threshold(s_hyb, all_, y, wte, x) for x in LOSS_LEVELS]]
    cost = te["subtree"].astype(np.float64) * wte
    for m, (name, s, el) in enumerate([("table", s_table, all_), ("MLP d<=6", s_mlp, shallow), ("hybride", s_hyb, all_)]):
        for j, x in enumerate(LOSS_LEVELS):
            sel = el & (s < thr[m][j])
            print(f"  mode {m + 1} {name:9s} perte {x:.1%} : seuil {thr[m][j]:+.3f}, "
                  f"coups vises {np.average(sel, weights=wte):.1%}, cout vise {cost[sel].sum() / cost.sum():.1%}")

    s = ck["state"]
    g = lambda k: s[k].double().numpy()  # noqa: E731
    with open(out, "w") as f:
        f.write("// Genere par tools/learned_pruning/quiet_export.py -- ne pas editer.\n#pragma once\n\nnamespace learned_quiet\n{\n")
        f.write(f"    constexpr int kMaxD = {MAX_D}, kMaxR = {MAX_R}, kHidden = {g('l0.weight').shape[0]}, "
                f"kX = {len(ck['mean'])}, kExperts = {qm.N_EXPERTS};\n")
        f.write(arr("kLogRate", log_rate.reshape(MAX_D + 1, MAX_R + 1)))
        f.write(arr("kLogCost", log_cost.reshape(MAX_D + 1, MAX_R + 1)))
        f.write(arr("kThr", thr))
        f.write(arr("kMean", ck["mean"]))
        f.write(arr("kStd", ck["std"]))
        f.write(arr("kL0WT", (g("l0.weight") / 127.0).T))
        f.write(arr("kL0B", g("l0.bias")))
        f.write(arr("kLinW", g("lin.weight")))
        f.write(arr("kLinB", g("lin.bias")))
        f.write(arr("kH1W", np.stack([g(f"h1.{i}.weight") for i in range(qm.N_EXPERTS)])))
        f.write(arr("kH1B", np.stack([g(f"h1.{i}.bias") for i in range(qm.N_EXPERTS)])))
        f.write(arr("kH2W", np.stack([g(f"h2.{i}.weight")[0] for i in range(qm.N_EXPERTS)])))
        f.write(arr("kH2B", np.array([g(f"h2.{i}.bias")[0] for i in range(qm.N_EXPERTS)])))
        f.write(arr("kPrior", ck["prior_logit"]))
        f.write("}\n")
    print(f"ecrit {out}")


def check(pt, dump, probs):
    ck = torch.load(pt, weights_only=False)
    r = np.fromfile(dump, dtype=qf.DTYPE)
    l0 = np.fromfile(dump + ".l0", dtype=np.uint8).reshape(-1, 1024)
    ze = np.fromfile(dump + ".z", dtype=np.float32)
    n = min(len(r), len(l0), len(ze))
    z = mlp_logit(ck, r[:n], l0[:n])
    err = np.abs(z - ze[:n])
    print(f"{n} coups, ecart max {err.max():.2e}, moyen {err.mean():.2e}")
    assert err.max() < 1e-3, "le moteur et quiet_mlp.py ne calculent pas le meme MLP"


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(*sys.argv[2:5])
    else:
        export(*sys.argv[1:4])
