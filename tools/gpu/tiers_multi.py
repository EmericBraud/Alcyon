"""Eval par paliers, plusieurs granularites a la fois (Matryoshka multi-k).

Une tete par taille k (nombre de neurones de tete de l'accumulateur, dans
l'ordre d'importance ft.perm ; prefixes emboites) + la tete complete. Tout est
entraine ensemble : perte complete + somme sur k de (perte cible + distillation
vers la complete). Suivi : perte complete et ecart de chaque palier a la
complete en cp (RMS, |eval| < 600).

Usage : python tiers_multi.py v5.pt ft.perm TRAIN VAL OUT --ks 32,64,128,256,512 --steps 30000
"""

import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, ".")
from model.config import LossParams  # noqa
from model.lightning_module import calculate_sf_loss  # noqa
from model.modules import LayerStacks  # noqa
from model.modules.config import LayerStacksConfig  # noqa
from tiers_probe import batches, wp  # noqa


def fwd(model, heads, idxs, b, fq):
    us, them, wi, bi, outcome, score, pc = b
    psqt_idx, ls_idx = model.calculate_buckets(pc)
    l0, wpsqt, bpsqt = model.forward_ft(us, them, wi, bi, psqt_idx, fq, fq)
    psqt = (wpsqt - bpsqt) * (us - 0.5)
    full = model.layer_stacks(l0, ls_idx, fq, fq) + psqt
    smalls = [h(l0[:, i], ls_idx, fq, fq) + psqt for h, i in zip(heads, idxs)]
    return full, smalls, outcome, score


@torch.no_grad()
def evaluate(model, heads, idxs, val, lp, lam, n2s):
    model.eval(); [h.eval() for h in heads]
    lf, d2, n = [], [0.0] * len(heads), 0
    for b in val:
        full, smalls, outcome, score = fwd(model, heads, idxs, b, False)
        lf.append(calculate_sf_loss(full * n2s, score, outcome, lp, lam).item())
        f = (full * n2s).squeeze(1)
        m = f.abs() < 600
        n += int(m.sum())
        for j, s in enumerate(smalls):
            d2[j] += (((s * n2s).squeeze(1) - f)[m] ** 2).sum().item()
    model.train(); [h.train() for h in heads]
    return float(np.mean(lf)), [math.sqrt(x / max(1, n)) for x in d2]


def main():
    ap = argparse.ArgumentParser()
    for x in ["model", "perm", "train", "val", "out"]:
        ap.add_argument(x)
    ap.add_argument("--ks", default="32,64,128,256,512")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--lr-model", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--w-small", type=float, default=0.25)
    ap.add_argument("--w-distill", type=float, default=5.0)
    ap.add_argument("--init-full", action="store_true")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--eval-every", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ks = [int(k) for k in a.ks.split(",")]

    obj = torch.load(a.model, weights_only=False)
    model = (obj.model if hasattr(obj, "model") else obj).cuda().train()
    q, n2s, half = model.quantization, model.quantization.nnue2score, model.L1 // 2
    order = torch.from_numpy(np.load(open(a.perm, "rb"))).cuda()
    idxs = [torch.cat([order[:k], order[:k] + half]) for k in ks]
    heads = [LayerStacks(model.num_ls_buckets, LayerStacksConfig(L1=2 * k, L2=model.L2, L3=model.L3), q).cuda().train() for k in ks]
    if a.init_full:
        full_sd = model.layer_stacks.state_dict()
        for h, i in zip(heads, idxs):
            sd = {}
            for name, t in full_sd.items():
                if name.startswith("l1.") and name.endswith("weight"):
                    t = t[:, i]
                sd[name] = t.clone()
            h.load_state_dict(sd)
    lp = LossParams()
    opt = torch.optim.Adam([{"params": model.parameters(), "lr": a.lr_model}] +
                           [{"params": h.parameters(), "lr": a.lr_head} for h in heads], eps=a.eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    gen = batches([a.train], a.bs, a.workers)
    val = [b for _, b in zip(range(30), batches([a.val], a.bs, a.workers))]

    lf0, _ = evaluate(model, heads, idxs, val, lp, a.lam, n2s)
    print(f"ks {ks} | depart perte complete {lf0:.6f}", flush=True)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        full, smalls, outcome, score = fwd(model, heads, idxs, next(gen), True)
        loss = calculate_sf_loss(full * n2s, score, outcome, lp, a.lam)
        tf = wp(full.detach() * n2s)
        for s in smalls:
            loss = loss + a.w_small * calculate_sf_loss(s * n2s, score, outcome, lp, a.lam) \
                        + a.w_distill * ((wp(s * n2s) - tf) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.clip_weights(True)
        if step % a.eval_every == 0 or step == a.steps:
            lf, rms = evaluate(model, heads, idxs, val, lp, a.lam, n2s)
            print(f"pas {step} ({time.time()-t0:.0f} s) : complete {lf:.6f} ({100*(lf/lf0-1):+.2f} %) | RMS cp " +
                  " ".join(f"k{k}={r:.1f}" for k, r in zip(ks, rms)), flush=True)
            torch.save({k: h.state_dict() for k, h in zip(ks, heads)}, os.path.join(a.out, "heads.pt"))
            if hasattr(obj, "model"): obj.model = model
            else: obj = model
            torch.save(obj, os.path.join(a.out, "v5m.pt"))
    print("termine", flush=True)


if __name__ == "__main__":
    main()
