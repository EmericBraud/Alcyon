"""Compare a checkpoint's predictions (v5 residual-correction, or a
legacy v1/v3/v4 standalone evaluator -- auto-detected) against the two
ground truths that actually matter: real game outcome (WDL) and the
binpack's own Stockfish searched score (whatever depth the dataset's
generator used) -- NOT Stockfish's static eval (that's
compare_checkpoints.py's wac.epd-based methodology, useful for
tactical blind-spot analysis but not the right target for "is this
model actually more accurate"). For a v5 checkpoint, also reports the
NNUE-alone baseline it's meant to correct, for direct comparison.

Uses the held-out validation split (--val-percent hash split, same one
used during training) so this can't be contaminated by training data.

WDL correlation is computed against tanh(logit / SCORE_SCALE) (matches
train.py's own loss target construction) since `result` is a WDL label
(-1/0/1-ish, see plane_batch.cpp) while the raw logit lives in a
different, unbounded space.

Reproducibility: the C++ stream has no RNG (no seed to fix), but with
num_workers > 1 several threads read disjoint binpack chunks in
parallel and interleave batches in whatever order they finish -- so
which ~49k-position sample you get is non-deterministic across runs.
Comparing two models measured in SEPARATE invocations can therefore
show apparent differences that are just sampling noise (empirically
+/-0.005-0.01 correlation on a 65536-position sample), not a real
quality difference -- seen when v3 vs v4 flipped WDL ranking between
two runs. Use --num-workers 1 (default) for a deterministic, repeatable
sample; only raise it if eval speed actually matters, and then treat
small cross-run deltas as noise, not signal -- prefer comparing
checkpoints within the SAME invocation (--checkpoints glob in
compare_checkpoints.py) or use a much larger --num-positions.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402

SCORE_SCALE = 410.0
MATE_ABS_THRESHOLD = 8000


def correlation(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-percent", type=int, default=5,
                         help="must match the --val-percent used during training")
    parser.add_argument("--num-positions", type=int, default=65536)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="binpack reader concurrency. 1 (default) gives a deterministic, "
        "repeatable sample -- see the module docstring on why >1 introduces "
        "run-to-run sampling noise in the comparison.",
    )
    parser.add_argument(
        "--nnue-calibration",
        default=None,
        help="Path to the NNUE calibration curve JSON -- defaults to the "
        "path recorded in the checkpoint itself (train.py saves it), if present.",
    )
    parser.add_argument(
        "--nnue-path",
        default=None,
        help="Path to the .nnue weight file -- defaults to the path recorded in "
        "the checkpoint itself, if present. Override when running on a machine "
        "with a different filesystem layout than where training happened.",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "mps", "cuda"],
        help="Model inference device. The data loader itself always runs on "
        "CPU (C++ stream); only the forward pass is moved to --device.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    model, is_v5, ckpt = load_model(args.checkpoint, device)

    nnue_path = (args.nnue_path or ckpt.get("nnue_path", "")) if is_v5 else ""
    dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.num_positions, cyclic=False,
        num_workers=args.num_workers, val_percent=args.val_percent, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, result, piece_count, nnue_score = next(iter(dataset))
    score, result, nnue_score = score.squeeze(-1), result.squeeze(-1), nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, result, piece_count, nnue_score = (
        planes[mask], score[mask], result[mask], piece_count[mask], nnue_score[mask]
    )
    print(f"sample: {planes.shape[0]} positions from the binpack's held-out validation split "
          f"(mate sentinels excluded)\n")

    planes = planes.to(device)
    piece_count = piece_count.to(device)
    nnue_score = nnue_score.to(device)
    score = score.to(device)
    result = result.to(device)

    num_planes = model.stem_conv.weight.shape[1]
    planes = planes[:, :num_planes]

    print(f"checkpoint step: {ckpt['step']}  (architecture: {'v5 residual-correction' if is_v5 else 'legacy standalone evaluator'})\n")

    if is_v5:
        calibration_path = args.nnue_calibration or ckpt.get("nnue_calibration")
        if calibration_path is None:
            raise SystemExit(
                f"{args.checkpoint} needs a calibration curve but --nnue-calibration was not "
                "given and none is recorded in the checkpoint."
            )
        calibration = NnueCalibration(calibration_path).to(device)
        nnue_logit = calibration(nnue_score)

        with torch.no_grad():
            logit = model(planes, piece_count, nnue_logit)

        nnue_wdl_pred = torch.tanh(nnue_logit / SCORE_SCALE)
        nnue_score_corr = correlation(nnue_logit * SCORE_SCALE, score)
        nnue_wdl_corr = correlation(nnue_wdl_pred, result)
    else:
        with torch.no_grad():
            logit = model(planes, piece_count)

    cnn_wdl_pred = torch.tanh(logit / SCORE_SCALE)
    cnn_score_corr = correlation(logit * SCORE_SCALE, score)
    cnn_wdl_corr = correlation(cnn_wdl_pred, result)

    print("vs Stockfish searched score (binpack's own 'score' field):")
    if is_v5:
        print(f"  NNUE alone:   corr={nnue_score_corr:.4f}")
    print(f"  model:        corr={cnn_score_corr:.4f}\n")
    print("vs real game outcome (WDL):")
    if is_v5:
        print(f"  NNUE alone:   corr={nnue_wdl_corr:.4f}")
    print(f"  model:        corr={cnn_wdl_corr:.4f}")


if __name__ == "__main__":
    main()
