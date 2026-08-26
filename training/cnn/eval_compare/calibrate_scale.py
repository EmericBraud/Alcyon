"""Calibrate the CNN's cp_scale on a large sample from the binpack's
held-out validation split (--val-percent hash split, same one used
during training) — completely independent of wac.epd, so it can't
contaminate the win-rate check in per_position_analysis.py.

Ground truth here is the binpack's own Stockfish score field (a
searched score, whatever depth the dataset's generator used) — a
different ground truth than wac.epd's static eval, but that's fine:
this step is purely about finding the right centipawn SCALE for the
CNN's output, not about measuring accuracy. Scale is a property of
the output distribution, learnable from any large enough sample of
(logit, true_cp) pairs.
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from model import ChessCNN  # noqa: E402

MATE_ABS_THRESHOLD = 8000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-percent", type=int, default=5,
                         help="must match the --val-percent used during training")
    parser.add_argument("--num-positions", type=int, default=16384)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"))
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ChessCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.num_positions, cyclic=False,
        num_workers=2, val_percent=args.val_percent, is_validation=True,
    )
    planes, score, result, piece_count = next(iter(dataset))
    score = score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, piece_count = planes[mask], score[mask], piece_count[mask]
    print(f"calibration sample: {planes.shape[0]} positions from the binpack's "
          f"validation split (mate sentinels excluded)")

    with torch.no_grad():
        logit = model(planes, piece_count)

    optimal_scale = (logit * score).sum() / (logit * logit).sum()
    residual_before = (410.0 * logit - score).abs().mean().item()
    residual_after = (optimal_scale * logit - score).abs().mean().item()

    print(f"default SCORE_SCALE=410.0  ->  MAE={residual_before:.1f} cp")
    print(f"calibrated scale={optimal_scale.item():.1f}  ->  MAE={residual_after:.1f} cp")

    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "checkpoint_step": ckpt["step"],
            "num_positions": planes.shape[0],
            "cp_scale": optimal_scale.item(),
        }, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
