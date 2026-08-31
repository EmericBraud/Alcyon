"""Test whether the NNUE's residual error (vs the searched Stockfish
score) has exploitable cluster structure, BEYOND what the phase
buckets already used by the CNN's value heads capture -- a
precondition for a mixture-of-experts gating architecture. If the
residual is roughly homogeneous once phase is accounted for, a MoE
gate learns nothing a single trunk with phase-bucketed heads doesn't
already get for free.

Method:
  1. Compute the NNUE's residual error in cp: nnue_cp - score
     (score = binpack's own Stockfish searched score).
  2. Build a small battery of simple, cheap-to-derive position
     features directly from the input planes (no CNN forward pass,
     no FEN re-parsing): phase (piece_count), |nnue score|, material
     presence (queens, pawn count), mobility proxies (attacked-square
     counts per side), king danger proxies (enemy attack coverage
     within 1 square of each king).
  3. Baseline: eta^2 (between-cluster variance / total variance of
     the residual) using the model's OWN phase buckets as "clusters"
     -- this is the structure already exploited by the existing
     per-bucket value heads.
  4. Candidate: eta^2 using k-means clusters fit on the OTHER features
     (phase excluded) -- structure a MoE gate could additionally
     exploit, if this eta^2 is meaningfully higher than the baseline.
  5. Also reports per-feature correlation with the residual, as a
     more interpretable (if partial) view of what's driving any
     structure found.
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


def correlation(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def eta_squared(residual, cluster_ids):
    """Fraction of residual variance explained by cluster membership
    (between-cluster SS / total SS) -- the standard ANOVA effect size."""
    total_ss = ((residual - residual.mean()) ** 2).sum()
    between_ss = torch.zeros(())
    for c in cluster_ids.unique():
        mask = cluster_ids == c
        if mask.sum() < 2:
            continue
        group_mean = residual[mask].mean()
        between_ss = between_ss + mask.sum() * (group_mean - residual.mean()) ** 2
    return (between_ss / (total_ss + 1e-9)).item()


def king_danger(planes, king_distance_plane_idx, enemy_attack_plane_range):
    """Sum of enemy-attack coverage on squares within 1 (Chebyshev) of
    the king -- king_distance planes are Chebyshev distance / 7, so
    "within 1" is distance <= 1/7 (i.e. the king's own square plus its
    8 neighbors)."""
    near_king = planes[:, king_distance_plane_idx] <= (1.0 / 7.0 + 1e-6)
    enemy_attack_union = planes[:, enemy_attack_plane_range[0]:enemy_attack_plane_range[1]].amax(dim=1)
    return (near_king * enemy_attack_union).sum(dim=(1, 2))


def build_features(planes, piece_count, nnue_cp):
    our_pawns = planes[:, 0].sum(dim=(1, 2))
    their_pawns = planes[:, 6].sum(dim=(1, 2))
    our_queen = (planes[:, 4].sum(dim=(1, 2)) > 0).float()
    their_queen = (planes[:, 10].sum(dim=(1, 2)) > 0).float()
    mobility_us = planes[:, 19:25].sum(dim=(1, 2, 3))
    mobility_them = planes[:, 25:31].sum(dim=(1, 2, 3))
    king_danger_us = king_danger(planes, 31, (25, 31))
    king_danger_them = king_danger(planes, 32, (19, 25))
    ep_present = (planes[:, 17].sum(dim=(1, 2)) > 0).float()

    features = {
        "abs_nnue_cp": nnue_cp.abs(),
        "total_pawns": our_pawns + their_pawns,
        "our_queen": our_queen,
        "their_queen": their_queen,
        "mobility_us": mobility_us,
        "mobility_them": mobility_them,
        "mobility_diff": mobility_us - mobility_them,
        "king_danger_us": king_danger_us,
        "king_danger_them": king_danger_them,
        "king_danger_diff": king_danger_us - king_danger_them,
        "ep_present": ep_present,
    }
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True, help="a v5 checkpoint, for its nnue_path/calibration")
    parser.add_argument("--val-percent", type=int, default=5)
    parser.add_argument("--num-positions", type=int, default=65536)
    parser.add_argument("--nnue-calibration", default=None)
    parser.add_argument("--nnue-path", default=None)
    parser.add_argument("--num-clusters", type=int, default=8,
                         help="k for the k-means clustering on non-phase features")
    args = parser.parse_args()

    device = torch.device("cpu")
    _, is_v5, ckpt = load_model(args.checkpoint, device)
    if not is_v5:
        raise SystemExit("residual_cluster_analysis.py needs a v5 checkpoint (for its nnue_path/calibration).")

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
    print(f"sample: {planes.shape[0]} positions from the binpack's held-out validation split\n")

    calibration_path = args.nnue_calibration or ckpt.get("nnue_calibration")
    calibration = NnueCalibration(calibration_path).to(device)
    nnue_logit = calibration(nnue_score)
    nnue_cp = nnue_logit * SCORE_SCALE

    residual = nnue_cp - score
    print(f"NNUE residual (nnue_cp - searched_score): mean={residual.mean():.1f}  "
          f"std={residual.std():.1f}\n")

    phase_bucket = phase_bucket_for_piece_count(piece_count)
    baseline_eta2 = eta_squared(residual, phase_bucket)
    print(f"Baseline (phase buckets, {phase_bucket.unique().numel()} groups, "
          f"already exploited by the model's own per-bucket heads):")
    print(f"  eta^2 = {baseline_eta2:.4f}\n")

    features = build_features(planes, piece_count, nnue_cp)
    print("Per-feature correlation with residual (interpretability, not causal):")
    for name, values in features.items():
        print(f"  {name:20s} corr={correlation(values, residual):+.4f}")
    print()

    from sklearn.cluster import KMeans
    import numpy as np

    feature_names = list(features.keys())
    X = torch.stack([features[n] for n in feature_names], dim=1).numpy()
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    kmeans = KMeans(n_clusters=args.num_clusters, n_init=10, random_state=0)
    cluster_ids = torch.from_numpy(kmeans.fit_predict(X))
    candidate_eta2 = eta_squared(residual, cluster_ids)
    print(f"Candidate (k-means, k={args.num_clusters}, on non-phase features above, "
          f"phase EXCLUDED -- structure a MoE gate could additionally exploit):")
    print(f"  eta^2 = {candidate_eta2:.4f}\n")

    # Combine phase bucket + kmeans cluster into a joint grouping, to see
    # if the kmeans clusters add anything ON TOP of phase (rather than
    # just rediscovering phase indirectly through piece-count-correlated
    # features like mobility/pawn count).
    joint_ids = phase_bucket * args.num_clusters + cluster_ids
    joint_eta2 = eta_squared(residual, joint_ids)
    incremental_eta2 = joint_eta2 - baseline_eta2
    print(f"Joint (phase bucket x k-means cluster):")
    print(f"  eta^2 = {joint_eta2:.4f}   (incremental over phase-only: {incremental_eta2:+.4f})\n")

    print("Verdict:")
    if incremental_eta2 < 0.02:
        print(f"  Incremental eta^2 ({incremental_eta2:.4f}) is small -- little evidence of "
              "residual cluster structure beyond what phase buckets already capture. "
              "A MoE gate would likely learn close to nothing extra over the current "
              "per-phase-bucket heads.")
    else:
        print(f"  Incremental eta^2 ({incremental_eta2:.4f}) is non-trivial -- there IS "
              "residual structure orthogonal to phase. Worth prototyping a soft MoE gate.")


if __name__ == "__main__":
    main()
