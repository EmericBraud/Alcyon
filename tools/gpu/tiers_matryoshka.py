"""Eval par paliers : fine-tuning "Matryoshka" de v5.

Tout le reseau est reentraine (accumulateur compris) avec deux tetes qui
partagent l'accumulateur : la tete complete (1024 entrees) et la petite tete
(les k neurones de tete de l'ordre d'importance, 2k entrees). Perte =
perte complete (cible des donnees, lambda de v5) + perte de la petite tete
(meme cible) + distillation de la petite tete vers la complete (pour garder
les paliers calibres entre eux). Le gradient de la petite tete pousse
l'accumulateur a concentrer l'information dans ses premiers neurones.

Suivi : perte de validation de la tete complete (ne doit pas se degrader par
rapport a v5) et ecart de la petite tete a la complete EN CENTIPIONS (RMS sur
les positions |eval| < 600 cp, la ou une erreur change des decisions).

Usage : python tiers_matryoshka.py v5.pt tier_head.pt ft.perm TRAIN VAL OUT_DIR --steps 8000
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from model.config import LossParams  # noqa: E402
from model.lightning_module import calculate_sf_loss  # noqa: E402
from model.modules import LayerStacks  # noqa: E402
from model.modules.config import LayerStacksConfig  # noqa: E402
from tiers_probe import batches, wp  # noqa: E402


def forward_both(model, head, idx, b, fq):
    us, them, wi, bi, outcome, score, pc = b
    psqt_idx, ls_idx = model.calculate_buckets(pc)
    l0, wpsqt, bpsqt = model.forward_ft(us, them, wi, bi, psqt_idx, fq, fq)
    psqt = (wpsqt - bpsqt) * (us - 0.5)
    full = model.layer_stacks(l0, ls_idx, fq, fq) + psqt
    small = head(l0[:, idx], ls_idx, fq, fq) + psqt
    return full, small, outcome, score


@torch.no_grad()
def evaluate(model, head, idx, val, lp, lam, n2s):
    model.eval()
    head.eval()
    lf, d2, n_mid, mae = [], 0.0, 0, []
    for b in val:
        full, small, outcome, score = forward_both(model, head, idx, b, False)
        lf.append(calculate_sf_loss(full * n2s, score, outcome, lp, lam).item())
        f_cp, s_cp = (full * n2s).squeeze(1), (small * n2s).squeeze(1)
        m = f_cp.abs() < 600
        d2 += ((s_cp - f_cp)[m] ** 2).sum().item()
        n_mid += int(m.sum())
        mae.append((s_cp - f_cp)[m].abs().mean().item())
    model.train()
    head.train()
    return float(np.mean(lf)), math.sqrt(d2 / max(1, n_mid)), float(np.mean(mae))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("head")
    ap.add_argument("perm")
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lr-model", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--w-small", type=float, default=1.0)
    ap.add_argument("--w-distill", type=float, default=20.0)
    ap.add_argument("--eval-every", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    obj = torch.load(a.model, weights_only=False)
    model = (obj.model if hasattr(obj, "model") else obj).cuda().train()
    q = model.quantization
    n2s = q.nnue2score
    half = model.L1 // 2
    with open(a.perm, "rb") as f:
        order = torch.from_numpy(np.load(f)).cuda()
    sel = order[: a.k]
    idx = torch.cat([sel, sel + half])
    head = LayerStacks(model.num_ls_buckets, LayerStacksConfig(L1=2 * a.k, L2=model.L2, L3=model.L3), q).cuda()
    head.load_state_dict(torch.load(a.head))
    head.train()
    lp = LossParams()

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": a.lr_model},
        {"params": head.parameters(), "lr": a.lr_head},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    gen = batches([a.train], a.bs, a.workers)
    val = [b for _, b in zip(range(30), batches([a.val], a.bs, a.workers))]

    lf0, rms0, mae0 = evaluate(model, head, idx, val, lp, a.lam, n2s)
    print(f"depart : perte complete {lf0:.6f} | petite tete vs complete : RMS {rms0:.1f} cp, MAE {mae0:.1f} cp (|eval| < 600)", flush=True)

    t0 = time.time()
    for step in range(1, a.steps + 1):
        full, small, outcome, score = forward_both(model, head, idx, next(gen), True)
        l_full = calculate_sf_loss(full * n2s, score, outcome, lp, a.lam)
        l_small = calculate_sf_loss(small * n2s, score, outcome, lp, a.lam)
        l_dist = ((wp(small * n2s) - wp(full.detach() * n2s)) ** 2).mean()
        loss = l_full + a.w_small * l_small + a.w_distill * l_dist
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        model.clip_weights(True)
        if step % a.eval_every == 0 or step == a.steps:
            lf, rms, mae = evaluate(model, head, idx, val, lp, a.lam, n2s)
            print(f"pas {step} ({time.time() - t0:.0f} s) : perte complete {lf:.6f} ({100 * (lf / lf0 - 1):+.2f} % vs v5) | "
                  f"petite tete RMS {rms:.1f} cp, MAE {mae:.1f} cp", flush=True)
            torch.save(head.state_dict(), os.path.join(a.out, "tier_head.pt"))
            if hasattr(obj, "model"):
                obj.model = model
            else:
                obj = model
            torch.save(obj, os.path.join(a.out, "v5m.pt"))
    print("termine", flush=True)


if __name__ == "__main__":
    main()
