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
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402

MATE_ABS_THRESHOLD = 8000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val-percent", type=int, default=5,
                         help="must match the --val-percent used during training")
    parser.add_argument("--num-positions", type=int, default=16384)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"))
    parser.add_argument(
        "--nnue-calibration",
        default=None,
        help="Path to the NNUE calibration curve JSON -- only required "
        "if --checkpoint is a v5 (residual-correction) one. Defaults to "
        "the path recorded in the checkpoint itself (train.py saves it), "
        "if present.",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    model, is_v5, ckpt = load_model(args.checkpoint, device)

    nnue_path = ckpt.get("nnue_path", "") if is_v5 else ""
    dataset = PlaneBatchDataset(
        filenames=args.binpack, batch_size=args.num_positions, cyclic=False,
        num_workers=2, val_percent=args.val_percent, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, result, piece_count, nnue_score = next(iter(dataset))
    score = score.squeeze(-1)
    nnue_score = nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, piece_count, nnue_score = planes[mask], score[mask], piece_count[mask], nnue_score[mask]
    print(f"calibration sample: {planes.shape[0]} positions from the binpack's "
          f"validation split (mate sentinels excluded)")

    # Older checkpoints (pre-v4) were trained with fewer input planes.
    num_planes = model.stem_conv.weight.shape[1]
    planes = planes[:, :num_planes]

    with torch.no_grad():
        if is_v5:
            calibration_path = args.nnue_calibration or ckpt.get("nnue_calibration")
            if calibration_path is None:
                raise SystemExit(
                    f"{args.checkpoint} is a v5 checkpoint (needs nnue_logit) but "
                    "--nnue-calibration was not given and none is recorded in the checkpoint."
                )
            calibration = NnueCalibration(calibration_path).to(device)
            nnue_logit = calibration(nnue_score)
            logit = model(planes, piece_count, nnue_logit)
        else:
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
