"""Calibrates chess26's own raw NNUE score into a logit, for v5's
residual-correction training (see model.py,
docs/gpu-async-eval/v5-hybrid-nnue-cnn.md).

Ground truth here is the binpack's own Stockfish `score` field, on the
held-out validation split (--val-percent hash split, same one used
during training) -- the SAME target train.py's SCORE_SCALE is
calibrated against, so the fitted curve puts nnue_logit on exactly the
logit scale the trunk's own output already lives in.

A single scalar scale (nnue_logit = nnue_score / NNUE_SCALE) is NOT
enough: measured on a 100k-position sample, the effective scale varies
by ~1.7x between near-equal positions (|nnue_score| small, effective
scale ~470) and decisive ones (|nnue_score| large, effective scale
~270-290) -- a real, magnitude-dependent non-linearity, not noise. A
single global scale would bias every position, and the CNN trunk has
no way to correct it itself (it never sees nnue_score as an input, so
it can't detect *when* the miscalibration applies). Instead, this
fits a monotonic piecewise-linear calibration curve (quantile-binned,
isotonic-style pooling to guarantee monotonicity) and saves it to a
JSON file (breakpoints_x, breakpoints_y) that train.py loads and
applies via linear interpolation (see nnue_calibration.py).

Do NOT calibrate against Stockfish's *static* eval (as
per_position_analysis.py's CNN-output calibration does) -- that's a
different quantity from the binpack's searched score field, and the
two scales don't transfer (measured a ~7x mismatch for the CNN's own
output calibration; expect something similar here).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402

MATE_ABS_THRESHOLD = 8000
SCORE_SCALE = 410.0  # must match train.py's SCORE_SCALE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--nnue-path", required=True)
    parser.add_argument("--val-percent", type=int, default=5,
                         help="must match the --val-percent used during training")
    parser.add_argument("--num-positions", type=int, default=200_000)
    parser.add_argument("--num-bins", type=int, default=64,
                         help="number of quantile bins for the piecewise-linear curve")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnue_calibration.json"),
    )
    args = parser.parse_args()

    dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.num_positions, cyclic=False,
        num_workers=4, val_percent=args.val_percent, is_validation=True,
        nnue_path=args.nnue_path,
    )
    _, score, _, _, nnue_score = next(iter(dataset))
    score = score.squeeze(-1)
    nnue_score = nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    score, nnue_score = score[mask], nnue_score[mask]
    target = score / SCORE_SCALE
    n = nnue_score.shape[0]
    print(f"calibration sample: {n} positions (mate sentinels excluded)")

    # Quantile bins (equal position count per bin, not equal nnue_score
    # width) so the tails -- fewer positions, more per-bin noise -- still
    # get statistically meaningful bin means.
    order = torch.argsort(nnue_score)
    nnue_sorted = nnue_score[order]
    target_sorted = target[order]

    bin_size = n // args.num_bins
    breakpoints_x, breakpoints_y = [], []
    for i in range(args.num_bins):
        lo = i * bin_size
        hi = n if i == args.num_bins - 1 else (i + 1) * bin_size
        breakpoints_x.append(nnue_sorted[lo:hi].mean().item())
        breakpoints_y.append(target_sorted[lo:hi].mean().item())

    # Isotonic-style pooling: enforce monotonicity in case sampling noise
    # (mostly at the sparse tails) makes a bin mean dip below the
    # previous one -- the true relationship must be monotonic (a more
    # decisive nnue_score should never map to a less decisive target).
    for i in range(1, len(breakpoints_y)):
        if breakpoints_y[i] < breakpoints_y[i - 1]:
            breakpoints_y[i] = breakpoints_y[i - 1]

    with open(args.out, "w") as f:
        json.dump({"x": breakpoints_x, "y": breakpoints_y}, f, indent=2)

    print("sample breakpoints (nnue_score -> target logit):")
    for i in range(0, args.num_bins, max(1, args.num_bins // 8)):
        print(f"  {breakpoints_x[i]:9.1f} -> {breakpoints_y[i]:.4f}")
    print(f"\nwrote {args.out}")
    print(f"pass this to train.py: --nnue-calibration {args.out}")


if __name__ == "__main__":
    main()
