"""Train a deliberately tiny, fast network to predict the NNUE's own
residual error directly (regression against nnue_cp - searched_score),
then check whether its learned embedding reveals cluster structure.

This is a stronger test than residual_cluster_analysis.py's linear
hand-crafted features (which found almost nothing) but a fairer one
than residual_cluster_embedding.py's use of the full v5 trunk's
embedding (which is circular -- v5 was trained to predict this exact
residual, so of course its embedding correlates with it, and it costs
as much as the whole model to compute anyway).

If THIS tiny net -- small enough to plausibly serve as a cheap MoE
gate -- can still (a) predict the residual with non-trivial accuracy
on held-out data and (b) show cluster structure beyond phase in its
embedding, that's real evidence a cheap gate is feasible. If it can't,
that's evidence the residual isn't cheaply predictable at all, which
undermines a sparse/fast MoE gate regardless of expert design.

Phase (piece_count) is deliberately NOT given as input -- we already
know phase explains almost nothing incremental (residual_cluster_
analysis.py), and we want to see if the net discovers OTHER structure
on its own from the raw planes, not just re-derive phase.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402
from model import phase_bucket_for_piece_count, NUM_PLANES  # noqa: E402

SCORE_SCALE = 410.0
MATE_ABS_THRESHOLD = 8000


class TinyBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class TinyResidualProbe(nn.Module):
    """Deliberately much smaller than v5's trunk (8x96, ~1.5M params):
    3 blocks x 24 channels, no squeeze-excitation, no phase-bucket
    heads -- a single linear regression head on top of a global-average
    -pooled embedding. Meant to be cheap enough to plausibly run as a
    per-position MoE gate."""

    def __init__(self, channels=24, num_blocks=3, num_planes=NUM_PLANES):
        super().__init__()
        self.stem_conv = nn.Conv2d(num_planes, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList(TinyBlock(channels) for _ in range(num_blocks))
        self.head = nn.Linear(channels, 1)

    def forward(self, planes, return_embedding=False):
        x = F.relu(self.stem_bn(self.stem_conv(planes)))
        for block in self.blocks:
            x = block(x)
        embedding = x.mean(dim=(2, 3))
        pred = self.head(embedding).squeeze(-1)
        if return_embedding:
            return pred, embedding
        return pred


def eta_squared(residual, cluster_ids):
    total_ss = ((residual - residual.mean()) ** 2).sum()
    between_ss = torch.zeros(())
    for c in cluster_ids.unique():
        mask = cluster_ids == c
        if mask.sum() < 2:
            continue
        group_mean = residual[mask].mean()
        between_ss = between_ss + mask.sum() * (group_mean - residual.mean()) ** 2
    return (between_ss / (total_ss + 1e-9)).item()


def correlation(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--v5-checkpoint", required=True, help="for its nnue_path/calibration only")
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-positions", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--nnue-path", default=None, help="override the path recorded in the checkpoint")
    parser.add_argument("--nnue-calibration", default=None, help="override the path recorded in the checkpoint")
    args = parser.parse_args()

    device = torch.device(args.device)
    _, is_v5, ckpt = load_model(args.v5_checkpoint, torch.device("cpu"))
    if not is_v5:
        raise SystemExit("--v5-checkpoint must be a v5 checkpoint (for its nnue_path/calibration).")
    nnue_path = args.nnue_path or ckpt.get("nnue_path", "")
    calibration_path = args.nnue_calibration or ckpt.get("nnue_calibration")
    calibration = NnueCalibration(calibration_path).to(device)

    probe = TinyResidualProbe(channels=args.channels, num_blocks=args.num_blocks).to(device)
    num_params = sum(p.numel() for p in probe.parameters())
    print(f"probe: {args.num_blocks} blocks x {args.channels} channels, {num_params:,} params "
          f"(v5's trunk: ~1.5M params, ~{1_500_000 // max(num_params, 1)}x bigger)\n")

    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    train_dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.batch_size, cyclic=True,
        num_workers=4, val_percent=args.val_percent, is_validation=False,
        nnue_path=nnue_path,
    )

    print(f"training {args.steps} steps, batch={args.batch_size}, device={args.device}...")
    start = time.perf_counter()
    train_iter = iter(train_dataset)
    for step in range(1, args.steps + 1):
        planes, score, _, piece_count, nnue_score = next(train_iter)
        planes, score, nnue_score = planes.to(device), score.to(device).squeeze(-1), nnue_score.to(device).squeeze(-1)

        mask = score.abs() < MATE_ABS_THRESHOLD
        planes, score, nnue_score = planes[mask], score[mask], nnue_score[mask]

        with torch.no_grad():
            nnue_cp = calibration(nnue_score) * SCORE_SCALE
            target_residual = nnue_cp - score

        pred_residual = probe(planes)
        loss = F.mse_loss(pred_residual, target_residual)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 250 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            print(f"  step {step}/{args.steps}  loss={loss.item():.1f}  ({elapsed:.0f}s elapsed)")

    print(f"\ntraining done in {time.perf_counter() - start:.0f}s\n")

    # --- Held-out evaluation ---
    val_dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.eval_positions, cyclic=False,
        num_workers=2, val_percent=args.val_percent, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, _, piece_count, nnue_score = next(iter(val_dataset))
    planes, score, nnue_score, piece_count = (
        planes.to(device), score.to(device).squeeze(-1), nnue_score.to(device).squeeze(-1), piece_count.to(device)
    )
    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, nnue_score, piece_count = planes[mask], score[mask], nnue_score[mask], piece_count[mask]
    print(f"eval sample: {planes.shape[0]} held-out validation positions\n")

    probe.eval()
    with torch.no_grad():
        nnue_cp = calibration(nnue_score) * SCORE_SCALE
        target_residual = (nnue_cp - score).cpu()
        pred_residual, embedding = probe(planes, return_embedding=True)
        pred_residual, embedding = pred_residual.cpu(), embedding.cpu()

    held_out_corr = correlation(pred_residual, target_residual)
    print(f"Held-out correlation (probe's predicted residual vs actual NNUE residual): {held_out_corr:.4f}")
    print("(0 = probe learned nothing useful about the residual; >0.3-ish = residual is meaningfully "
          "predictable even by this tiny net)\n")

    # --- Gate-effectiveness analysis ---
    # Can a cheap |predicted residual| threshold identify which positions
    # are worth running the full v5 correction on? Sort positions by the
    # probe's predicted |residual|, and for various "run correction on the
    # top X%" strategies, measure what fraction of the TOTAL actual |
    # residual| mass gets captured -- i.e. how much of the real correction
    # need is concentrated in the positions the cheap gate flags.
    abs_target = target_residual.abs()
    abs_pred = pred_residual.abs()
    total_abs_residual = abs_target.sum()
    order = torch.argsort(abs_pred, descending=True)
    sorted_target = abs_target[order]
    cum_captured = torch.cumsum(sorted_target, dim=0) / total_abs_residual

    print("Gate-effectiveness: running v5's correction on only the top X% of positions")
    print("(ranked by the probe's predicted |residual|) captures this fraction of the")
    print("TOTAL actual |residual| mass in the validation set (higher = better gate):\n")
    n = abs_target.shape[0]
    for frac in (0.10, 0.20, 0.30, 0.50, 0.70):
        idx = int(frac * n) - 1
        print(f"  top {frac*100:4.0f}% by predicted magnitude  ->  captures {cum_captured[idx]*100:5.1f}% "
              f"of total |residual|  (random baseline: {frac*100:5.1f}%)")
    print()

    phase_bucket = phase_bucket_for_piece_count(piece_count).cpu()
    baseline_eta2 = eta_squared(target_residual, phase_bucket)
    print(f"Baseline (phase buckets): eta^2 = {baseline_eta2:.4f}\n")

    from sklearn.cluster import KMeans

    X = embedding.numpy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    for k in (4, 8, 16, 32):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        cluster_ids = torch.from_numpy(kmeans.fit_predict(X))
        candidate_eta2 = eta_squared(target_residual, cluster_ids)
        joint_ids = phase_bucket * k + cluster_ids
        joint_eta2 = eta_squared(target_residual, joint_ids)
        incremental = joint_eta2 - baseline_eta2
        print(f"k={k:3d}  tiny-probe-embedding-cluster eta^2={candidate_eta2:.4f}  "
              f"joint eta^2={joint_eta2:.4f}  incremental over phase={incremental:+.4f}")


if __name__ == "__main__":
    main()
