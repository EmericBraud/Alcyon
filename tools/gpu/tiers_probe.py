"""Eval par paliers : precision d'une petite tete sur un sous-ensemble de l'accumulateur.

v5 reste FIGE. On calcule son l0 (1024 = 2 perspectives x 512 produits par
paires) et sa sortie, puis on entraine une petite pile de couches (memes
L2/L3, memes 8 buckets) qui ne lit que k neurones par perspective, c'est-a-dire
2k entrees de l0, soit 2k entrees d'accumulateur par perspective a mettre a
jour (le produit par paires lit i et i+512). Cible : la sortie du grand reseau (distillation),
pour que les paliers restent calibres entre eux.

Deux facons de choisir les k neurones : les k premiers (prefixe brut) ou les k
plus importants (activation moyenne x norme des poids sortants), ce qui
revient a permuter le FT avant de prendre un prefixe.

Usage : python tiers_probe.py v5.pt TRAIN.binpack VAL.binpack --k 128 64 256
"""

import argparse
import math
import sys
import time

import torch

sys.path.insert(0, ".")
import data_loader  # noqa: E402
from model.modules import LayerStacks  # noqa: E402
from model.modules.config import LayerStacksConfig  # noqa: E402

FEATURES = "Full_Threats+HalfKAv2_hm"  # nom cote chargeur (le ^ ne concerne que le modele)


def batches(files, bs, workers):
    ds = data_loader.SparseBatchDataset(FEATURES, files, bs, num_workers=workers)
    for b in iter(ds):
        yield [t.cuda(non_blocking=True) for t in b]


@torch.no_grad()
def full_pass(model, b):
    us, them, wi, bi, outcome, score, pc = b
    psqt_idx, ls_idx = model.calculate_buckets(pc)
    l0, wpsqt, bpsqt = model.forward_ft(us, them, wi, bi, psqt_idx, False, False)
    psqt = (wpsqt - bpsqt) * (us - 0.5)
    out = model.layer_stacks(l0, ls_idx, False, False) + psqt
    return l0, psqt, ls_idx, out, score


def wp(cp):
    # probabilite de gain approchee : compare les erreurs a une echelle qui compte
    return torch.sigmoid(cp / 400.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("--k", type=int, nargs="+", default=[128])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--val-batches", type=int, default=40)
    a = ap.parse_args()

    obj = torch.load(a.model, weights_only=False)
    model = (obj.model if hasattr(obj, "model") else obj).cuda().eval()
    q = model.quantization
    n2s = q.nnue2score
    half = model.L1 // 2

    # Importance des neurones : activation moyenne x poids sortants de l1 (les
    # deux perspectives partagent le meme indice de neurone).
    gen = batches([a.train], a.bs, a.workers)
    act = torch.zeros(model.L1, device="cuda")
    for _ in range(10):
        l0, *_ = full_pass(model, next(gen))
        act += l0.abs().mean(0)
    w = model.layer_stacks.l1.linear.weight.detach().abs().sum(0)  # [L1]
    imp = (act * w)
    imp = imp[:half] + imp[half:]
    order = torch.argsort(imp, descending=True)

    val = [full_pass(model, b) for _, b in zip(range(a.val_batches), batches([a.val], a.bs, a.workers))]
    print(f"validation : {a.val_batches * a.bs} positions", flush=True)

    for k in a.k:
        for mode in ("importance", "prefixe"):
            sel = order[:k] if mode == "importance" else torch.arange(k, device="cuda")
            idx = torch.cat([sel, sel + half])
            head = LayerStacks(model.num_ls_buckets, LayerStacksConfig(L1=2 * k, L2=model.L2, L3=model.L3), q).cuda()
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
            t0 = time.time()
            for step in range(a.steps):
                l0, psqt, ls, out, _ = full_pass(model, next(gen))
                pred = head(l0[:, idx], ls, False, False) + psqt
                loss = ((wp(pred * n2s) - wp(out * n2s)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            with torch.no_grad():
                d_wp, d_cp, base = [], [], []
                for l0, psqt, ls, out, score in val:
                    pred = head(l0[:, idx], ls, False, False) + psqt
                    d_wp.append(((wp(pred * n2s) - wp(out * n2s)) ** 2).mean())
                    d_cp.append((pred - out).abs().mean() * n2s)
                    # reference : ecart du grand reseau a la cible des donnees
                    base.append(((wp(out * n2s) - wp(score.float())) ** 2).mean())
                mse = torch.stack(d_wp).mean().item()
                cp = torch.stack(d_cp).mean().item()
                ref = torch.stack(base).mean().item()
            print(f"k={k:3d} ({2 * k:4d} entrees d'accumulateur sur {model.L1} par perspective) {mode:10s} : "
                  f"ecart au grand reseau {cp:6.1f} cp moyen, MSE proba {mse:.5f} "
                  f"(grand reseau vs donnees : {ref:.5f}) en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
