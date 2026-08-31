"""Stronger version of residual_cluster_analysis.py's hypothesis test:
instead of hand-crafted linear features (mobility counts, king danger
proxies, ...), cluster on the v5 trunk's OWN learned embedding
(pooled_final, the 96-dim vector just before the value head) --
whatever non-linear position structure a CNN can represent is present
in that space, so this is a much harder test for the null hypothesis
("NNUE's residual error has no exploitable cluster structure") than
simple linear features can ever be.

Still measures eta^2 (variance of NNUE's residual explained by cluster
membership) against the same phase-bucket baseline as
residual_cluster_analysis.py, so the two are directly comparable.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402
from model import phase_bucket_for_piece_count  # noqa: E402

SCORE_SCALE = 410.0
MATE_ABS_THRESHOLD = 8000


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--num-positions", type=int, default=65536)
    parser.add_argument("--nnue-calibration", default=None)
    parser.add_argument("--nnue-path", default=None)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model, is_v5, ckpt = load_model(args.checkpoint, device)
    if not is_v5:
        raise SystemExit("residual_cluster_embedding.py needs a v5 checkpoint.")

    nnue_path = args.nnue_path or ckpt.get("nnue_path", "")
    dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.num_positions, cyclic=False,
        num_workers=2, val_percent=args.val_percent, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, result, piece_count, nnue_score = next(iter(dataset))
    score, result, nnue_score = score.squeeze(-1), result.squeeze(-1), nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, result, piece_count, nnue_score = (
        planes[mask], score[mask], result[mask], piece_count[mask], nnue_score[mask]
    )
    planes, piece_count, nnue_score = planes.to(device), piece_count.to(device), nnue_score.to(device)
    print(f"sample: {planes.shape[0]} positions from the binpack's held-out validation split\n")

    calibration_path = args.nnue_calibration or ckpt.get("nnue_calibration")
    calibration = NnueCalibration(calibration_path).to(device)
    nnue_logit = calibration(nnue_score)
    nnue_cp = (nnue_logit * SCORE_SCALE).cpu()
    residual = nnue_cp - score

    print(f"NNUE residual (nnue_cp - searched_score): mean={residual.mean():.1f}  "
          f"std={residual.std():.1f}\n")

    # Grab the trunk's own learned embedding via a forward hook on the
    # last residual block -- pooled_final in model.py's forward(), but
    # not exposed as a return value, so we recompute the same mean-pool
    # here instead of duplicating the whole forward pass by hand.
    captured = {}

    def hook(module, inp, out):
        captured["last_block_out"] = out

    handle = model.blocks[-1].register_forward_hook(hook)
    with torch.no_grad():
        model(planes, piece_count, nnue_logit)
    handle.remove()
    pooled_final = captured["last_block_out"].mean(dim=(2, 3)).cpu()  # (N, channels)
    print(f"embedding shape: {tuple(pooled_final.shape)}\n")

    phase_bucket = phase_bucket_for_piece_count(piece_count).cpu()
    baseline_eta2 = eta_squared(residual, phase_bucket)
    print(f"Baseline (phase buckets, {phase_bucket.unique().numel()} groups):")
    print(f"  eta^2 = {baseline_eta2:.4f}\n")

    from sklearn.cluster import KMeans
    import numpy as np

    X = pooled_final.numpy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    for k in (4, 8, 16, 32):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
        cluster_ids = torch.from_numpy(kmeans.fit_predict(X))
        candidate_eta2 = eta_squared(residual, cluster_ids)
        joint_ids = phase_bucket * k + cluster_ids
        joint_eta2 = eta_squared(residual, joint_ids)
        incremental = joint_eta2 - baseline_eta2
        print(f"k={k:3d}  embedding-cluster eta^2={candidate_eta2:.4f}  "
              f"joint(phase x cluster) eta^2={joint_eta2:.4f}  "
              f"incremental over phase={incremental:+.4f}")


if __name__ == "__main__":
    main()
