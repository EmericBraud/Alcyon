"""Eval par paliers : produit tout ce qu'il faut au moteur.

1. ordre d'importance des 512 neurones (activation moyenne x poids sortants) ;
2. petite tete entrainee par distillation de v5 (fige) sur les k premiers
   neurones de cet ordre ;
3. permutation enregistree au format de ftperm.py (le neurone i du reseau
   permute est l'ancien neurone perm[i]) : `serialize.py --ft-perm` donne un
   v5 permute, de sortie identique, dont le palier reduit est un prefixe ;
4. en-tete C++ des poids flottants de la petite tete (couches regroupees par
   bucket, facteur inclus), avec un auto-test : la propagation numpy des poids
   exportes doit redonner la sortie torch.

Usage : python tiers_export.py v5.pt TRAIN.binpack VAL.binpack OUT_DIR --k 128 --steps 8000
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from model.modules import LayerStacks  # noqa: E402
from model.modules.config import LayerStacksConfig  # noqa: E402
from tiers_probe import batches, full_pass, wp  # noqa: E402


def arr(name, a):
    a = np.asarray(a, dtype=np.float32).ravel()
    return f"inline constexpr float {name}[{a.size}] = {{{','.join(f'{x:.8e}f' for x in a)}}};"


def np_forward(x, l1w, l1b, l2w, l2b, ow, ob, max_act):
    """Propagation flottante identique a LayerStacks.forward sans quantification."""
    l1c = x @ l1w.T + l1b
    skip = l1c[:, -2:-1] - l1c[:, -1:]
    l1x = np.clip(np.concatenate([l1c ** 2, l1c], 1), 0, max_act)
    l2c = l1x @ l2w.T + l2b
    l2x = np.clip(np.concatenate([l2c ** 2, l2c], 1), 0, max_act)
    return np.concatenate([l1x, l2x], 1) @ ow.T + ob + skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("train")
    ap.add_argument("val")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--load-head", default=None, help="reprend une tete deja entrainee (tier_head.pt) sans la reentrainer")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    obj = torch.load(a.model, weights_only=False)
    model = (obj.model if hasattr(obj, "model") else obj).cuda().eval()
    q = model.quantization
    half = model.L1 // 2
    assert abs(q.sqr_crelu_correction_factor - 1.0) < 1e-9, "export suppose une correction sqr-crelu de 1"

    gen = batches([a.train], a.bs, a.workers)
    act = torch.zeros(model.L1, device="cuda")
    for _ in range(20):
        l0, *_ = full_pass(model, next(gen))
        act += l0.abs().mean(0)
    w = model.layer_stacks.l1.linear.weight.detach().abs().sum(0)
    imp = act * w
    order = torch.argsort(imp[:half] + imp[half:], descending=True)
    np.save(os.path.join(a.out, "ft.perm"), order.cpu().numpy().astype(np.int64))
    os.replace(os.path.join(a.out, "ft.perm.npy"), os.path.join(a.out, "ft.perm"))

    k = a.k
    sel = order[:k]
    idx = torch.cat([sel, sel + half])
    head = LayerStacks(model.num_ls_buckets, LayerStacksConfig(L1=2 * k, L2=model.L2, L3=model.L3), q).cuda()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.steps))
    if a.load_head:
        head.load_state_dict(torch.load(a.load_head))
        a.steps = 0
    for step in range(a.steps):
        l0, psqt, ls, out, _ = full_pass(model, next(gen))
        pred = head(l0[:, idx], ls, False, False) + psqt
        loss = ((wp(pred * q.nnue2score) - wp(out * q.nnue2score)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            print(f"pas {step} perte {loss.item():.5f}", flush=True)

    head.eval()
    val = [full_pass(model, b) for _, b in zip(range(40), batches([a.val], a.bs, a.workers))]
    with torch.no_grad():
        mse = torch.stack([((wp((head(l0[:, idx], ls, False, False) + p) * q.nnue2score) - wp(o * q.nnue2score)) ** 2).mean()
                           for l0, p, ls, o, _ in val]).mean().item()
    print(f"k={k} : MSE proba vs v5 = {mse:.5f}", flush=True)

    torch.save(head.state_dict(), os.path.join(a.out, "tier_head.pt"))

    # Auto-test : reference torch calculee sur GPU avant de passer la tete sur
    # CPU (get_coalesced_layer_stacks construit ses couches sur CPU).
    l0, psqt, ls, out, _ = val[0]
    with torch.no_grad():
        ref = head(l0[:, idx], ls, False, False).squeeze(1).cpu().numpy()
    head = head.cpu()

    # Poids regroupes par bucket (facteur inclus).
    L1w, L1b, L2w, L2b, Ow, Ob = [], [], [], [], [], []
    for l1, l2, o in head.get_coalesced_layer_stacks():
        L1w.append(l1.weight.detach().cpu().numpy()); L1b.append(l1.bias.detach().cpu().numpy())
        L2w.append(l2.weight.detach().cpu().numpy()); L2b.append(l2.bias.detach().cpu().numpy())
        Ow.append(o.weight.detach().cpu().numpy()); Ob.append(o.bias.detach().cpu().numpy())
    max_act = q.max_hidden_activation

    # Auto-test sur un lot : numpy par bucket == torch.
    x = l0[:, idx].cpu().numpy()
    lsn = ls.cpu().numpy()
    got = np.empty_like(ref)
    for b in range(model.num_ls_buckets):
        m = lsn == b
        if m.any():
            got[m] = np_forward(x[m], L1w[b], L1b[b], L2w[b], L2b[b], Ow[b], Ob[b], max_act)[:, 0]
    err = np.abs(got - ref).max()
    print(f"auto-test export : ecart max numpy/torch = {err:.2e}", flush=True)
    assert err < 1e-3

    hdr = [
        "// Genere par tools/gpu/tiers_export.py : petite tete du palier reduit",
        "// (distillation de v5, entree = l0 / 128 des k premiers neurones du reseau",
        "// permute, deux perspectives). Ne pas editer a la main.",
        "#pragma once",
        "namespace tier_head",
        "{",
        f"inline constexpr int kK = {k};",
        f"inline constexpr int kIn = {2 * k};",
        f"inline constexpr int kL2 = {model.L2};",
        f"inline constexpr int kL3 = {model.L3};",
        f"inline constexpr int kBuckets = {model.num_ls_buckets};",
        f"inline constexpr float kMaxAct = {max_act:.8e}f;",
        f"inline constexpr float kNnue2Score = {q.nnue2score:.8e}f;",
        arr("kL1W", np.stack(L1w)), arr("kL1B", np.stack(L1b)),
        arr("kL2W", np.stack(L2w)), arr("kL2B", np.stack(L2b)),
        arr("kOW", np.stack(Ow)), arr("kOB", np.stack(Ob)),
        "}",
    ]
    with open(os.path.join(a.out, "tier_head_weights.hpp"), "w") as f:
        f.write("\n".join(hdr) + "\n")
    print("export termine", flush=True)


if __name__ == "__main__":
    main()
