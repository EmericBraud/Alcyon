"""Is v3's (standalone) error more or less correlated with NNUE's error
than v6's (residual-correction) error is?

Intuition check: v6 = nnue_logit + trunk_logits, so whatever NNUE-error
the trunk fails to cancel remains additively embedded in v6's own
error -- a direct architectural link. v3 has no such link (trained
independently, never sees the NNUE score). So *a priori* v6's error
should be MORE correlated with NNUE's error than v3's, not less --
this script checks that directly rather than guessing.

Ground truth: the binpack's own Stockfish searched score (continuous,
less noisy than WDL) -- matches the error-correlation diagnostic
methodology from experiment-v1.md (per_position_analysis.py).
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402

SCORE_SCALE = 410.0
MATE_ABS_THRESHOLD = 8000
NUM_POSITIONS = 65536


def correlation(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    binpack = os.path.join(here, "..", "..", "datasets", "test80-2024-01-jan-2tb7p.min-v2.v6.binpack")
    nnue_path = os.path.join(here, "..", "..", "..", "data", "nnue", "v3.nnue")
    v3_ckpt = os.path.join(here, "..", "checkpoints", "chesscnn_v3_step300000.pt")
    v6_ckpt = os.path.join(here, "..", "checkpoints_v6", "chesscnn_step200000.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Single dataset read (num_workers=1, deterministic) shared by both
    # models -- same validation sample for a fair error-correlation
    # comparison. See eval_compare_sampling_noise memory.
    dataset = PlaneBatchDataset(
        filenames=[binpack], batch_size=NUM_POSITIONS, cyclic=False,
        num_workers=1, val_percent=5, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, result, piece_count, nnue_score = next(iter(dataset))
    score, nnue_score = score.squeeze(-1), nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, piece_count, nnue_score = planes[mask], score[mask], piece_count[mask], nnue_score[mask]
    planes, score, piece_count, nnue_score = (
        planes.to(device), score.to(device), piece_count.to(device), nnue_score.to(device)
    )
    print(f"sample: {planes.shape[0]} positions, device={device}\n")

    calibration = NnueCalibration(os.path.join(here, "nnue_calibration.json")).to(device)
    nnue_logit = calibration(nnue_score)
    nnue_cp = nnue_logit * SCORE_SCALE
    nnue_error = nnue_cp - score

    v3_model, is_v5_v3, _ = load_model(v3_ckpt, device)
    assert not is_v5_v3
    v3_planes = planes[:, :v3_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v3_logit = v3_model(v3_planes, piece_count)
    v3_cp = v3_logit * SCORE_SCALE
    v3_error = v3_cp - score

    v6_model, is_v5_v6, ckpt6 = load_model(v6_ckpt, device)
    assert is_v5_v6
    v6_planes = planes[:, :v6_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v6_logit = v6_model(v6_planes, piece_count, nnue_logit)
    v6_cp = v6_logit * SCORE_SCALE
    v6_error = v6_cp - score

    print(f"NNUE error:  mean={nnue_error.mean():+.1f}  std={nnue_error.std():.1f}")
    print(f"v3 error:    mean={v3_error.mean():+.1f}  std={v3_error.std():.1f}")
    print(f"v6 error:    mean={v6_error.mean():+.1f}  std={v6_error.std():.1f}\n")

    corr_nnue_v3 = correlation(nnue_error, v3_error)
    corr_nnue_v6 = correlation(nnue_error, v6_error)
    corr_v3_v6 = correlation(v3_error, v6_error)

    print(f"corr(NNUE error, v3 error) = {corr_nnue_v3:.4f}")
    print(f"corr(NNUE error, v6 error) = {corr_nnue_v6:.4f}")
    print(f"corr(v3 error,   v6 error) = {corr_v3_v6:.4f}\n")

    if corr_nnue_v6 > corr_nnue_v3:
        print("v6's error IS more correlated with NNUE's error than v3's is "
              "(as the architectural link would predict).")
    else:
        print("v3's error is (surprisingly) more correlated with NNUE's error "
              "than v6's is -- against the naive architectural-link prediction.")


if __name__ == "__main__":
    main()
