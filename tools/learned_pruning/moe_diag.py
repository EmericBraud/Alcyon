"""Diagnostic des modeles de train_safe.py : effectifs et loss par bucket.

    python3 moe_diag.py <train.bin> <test.bin> <R> model_none.pt model_ds.pt model_dsp.pt

Par bucket du schema dsp (profondeur x cote du reduit x phase) : effectif en
train (apres le plafond de train_mlp.subsample) et en test, logloss train /
test de chaque modele. Puis taux d'erreur du reduit et logloss par nombre
exact de pieces : dit si la phase merite un decoupage plus fin.
"""
import sys

import numpy as np
import torch

import train_mlp
import train_safe

DEPTH_NAMES = ["1", "2", "3", "4-6", "7+"]


def predict(path, r, l0, idx, k, mean, std):
    ck = torch.load(path, weights_only=False)
    scheme = ck.get("moe", "none")
    d, x, p, y, _, _, red_fh, full_fh, b = train_safe.tensors(r, l0, idx, k, ck["mean"], ck["std"], scheme)
    if scheme == "none":
        net = train_mlp.Net(d.shape[1], True, ck["hidden"])
        net.load_state_dict(ck["state"])
        f = lambda i: net(d[i], x[i], p[i])  # noqa: E731
    else:
        n_b = train_safe.buckets(r[idx[:1]], np.zeros(1, bool), scheme)[1]
        net = train_safe.MoENet(d.shape[1], ck["hidden"], n_b)
        net.load_state_dict(ck["state"])
        f = lambda i: net(d[i], x[i], p[i], b[i])  # noqa: E731
    with torch.no_grad():
        out = torch.cat([torch.sigmoid(f(slice(i, i + 8192))) for i in range(0, len(d), 8192)]).numpy()
    return out, y.numpy()


def ll(p, y):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    tr_path, te_path, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
    models = sys.argv[4:]
    k = R - 1
    rtr, l0tr, itr = train_mlp.load(tr_path)
    rte, l0te, ite = train_mlp.load(te_path)
    itr = train_mlp.subsample(rtr, itr, np.random.default_rng(0))
    # Train evalue sur 1M noeuds tires du sous-echantillon d'entrainement.
    itr_ev = np.sort(np.random.default_rng(2).choice(itr, min(len(itr), 1_000_000), replace=False))
    red_tr = rtr["red_score"][itr, k] >= rtr["beta"][itr]
    red_te = rte["red_score"][ite, k] >= rte["beta"][ite]
    b_tr, _ = train_safe.buckets(rtr[itr], red_tr, "dsp")
    b_te, _ = train_safe.buckets(rte[ite], red_te, "dsp")
    red_ev = rtr["red_score"][itr_ev, k] >= rtr["beta"][itr_ev]
    b_ev, _ = train_safe.buckets(rtr[itr_ev], red_ev, "dsp")
    preds_te, preds_ev = {}, {}
    for m in models:
        name = torch.load(m, weights_only=False).get("moe", "none")
        preds_te[name], y_te = predict(m, rte, l0te, ite, k, None, None)
        preds_ev[name], y_ev = predict(m, rtr, l0tr, itr_ev, k, None, None)
    names = list(preds_te)
    print(f"R={R} ; effectifs et logloss par bucket (train sur 1M noeuds du sous-echantillon)\n")
    head = " ".join(f"{n + ' tr/te':>15s}" for n in names)
    print(f"{'depth':>5s} {'reduit':>6s} {'phase':>7s} {'n train':>9s} {'n test':>8s} {'erreurs':>8s}  {head}")
    for b in range(20):
        depth, side, phase = b // 4, (b // 2) % 2, b % 2
        st, se, sv = b_tr == b, b_te == b, b_ev == b
        cells = " ".join(f"{ll(preds_ev[n][sv], y_ev[sv]):7.4f}/{ll(preds_te[n][se], y_te[se]):.4f}" if se.sum() and sv.sum() else f"{'--':>15s}" for n in names)
        err = 1 - y_te[se].mean() if se.any() else float("nan")
        print(f"{DEPTH_NAMES[depth]:>5s} {'FH' if side else 'FL':>6s} {'>12' if phase else '<=12':>7s} {st.sum():9d} {se.sum():8d} {err:8.2%}  {cells}")
    pieces = rte["us"][ite].sum(1) + rte["them"][ite].sum(1) + 2
    print(f"\npar nombre de pieces (test) : effectif, erreurs du reduit, logloss " + " / ".join(names))
    for lo, hi in ((2, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 16), (17, 20), (21, 24), (25, 28), (29, 32)):
        s = (pieces >= lo) & (pieces <= hi)
        if s.sum() < 1000:
            continue
        print(f"  {lo:2d}-{hi:2d} pieces : {s.sum():8d}  {1 - y_te[s].mean():6.2%}  " + " / ".join(f"{ll(preds_te[n][s], y_te[s]):.4f}" for n in names))


if __name__ == "__main__":
    main()
