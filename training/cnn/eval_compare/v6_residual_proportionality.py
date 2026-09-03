"""Is v6's predicted correction actually proportional to NNUE's real
error vs. ground truth?

v6 is trained as score_final = nnue_logit + trunk_correction, targeting
trunk_correction ~= target - nnue_logit (see docs/gpu-async-eval/
v5-hybrid-nnue-cnn.md). If that training actually worked, the predicted
correction (v6 - nnue) should track the REAL residual NNUE needs fixed
(ground_truth - nnue) -- same sign at least, ideally close to a 1:1
slope. This checks that directly instead of assuming the training
objective was achieved.

Run in BOTH spaces:
- Stockfish searched score (raw cp, via SCORE_SCALE=410) -- but this is
  scale-sensitive: decisive positions naturally have far more cp
  headroom for "error" than balanced ones, which can distort mean/
  correlation comparisons (a model can be "wrong by 300cp" on a
  position that's winning either way -- not a meaningful error for what
  actually matters, the eventual result).
- WDL (real game outcome, tanh(logit / SCORE_SCALE) as win-probability,
  matching train.py's own loss target and compare_vs_binpack.py's
  methodology) -- compresses decisive positions instead of amplifying
  them, so this is the more meaningful space to answer "is v6's
  correction actually informative about what matters" in.

Same held-out test split / normalization pipeline as
error_correlation_v3_v6.py, disagreement_correlation.py and
compare_vs_binpack.py (calibrated NNUE logit + SCORE_SCALE=410, the
binpack's own Stockfish score AND its WDL result field as the two
ground truths).
"""

import os
import sys

import torch
from scipy.stats import kendalltau, spearmanr

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


def analyze(space_name, ground_truth, nnue_pred, v6_pred):
    """ground_truth/nnue_pred/v6_pred: same-scale tensors (either cp or
    WDL win-probability, see call sites)."""
    real_residual = ground_truth - nnue_pred
    predicted_residual = v6_pred - nnue_pred

    pearson_corr = correlation(predicted_residual, real_residual)
    pr_np = predicted_residual.detach().cpu().numpy()
    rr_np = real_residual.detach().cpu().numpy()
    spearman_corr, spearman_p = spearmanr(pr_np, rr_np)
    kendall_corr, kendall_p = kendalltau(pr_np, rr_np)

    pr_c = predicted_residual - predicted_residual.mean()
    rr_c = real_residual - real_residual.mean()
    slope = (pr_c * rr_c).sum() / (pr_c * pr_c).sum()

    same_sign = (predicted_residual.sign() == real_residual.sign())

    nnue_abs_error = real_residual.abs()
    v6_abs_error = (ground_truth - v6_pred).abs()
    improved = (v6_abs_error < nnue_abs_error)

    constant_shift = real_residual.mean()
    constant_abs_error = (real_residual - constant_shift).abs()

    print(f"=== {space_name} ===")
    print(f"real residual (truth - nnue):      mean={real_residual.mean():+.4g}  std={real_residual.std():.4g}")
    print(f"predicted residual (v6 - nnue):     mean={predicted_residual.mean():+.4g}  std={predicted_residual.std():.4g}")
    print(f"Pearson  corr(predicted, real) = {pearson_corr:.4f}")
    print(f"Spearman corr(predicted, real) = {spearman_corr:.4f}  (p={spearman_p:.2e})")
    print(f"Kendall  tau(predicted, real)  = {kendall_corr:.4f}  (p={kendall_p:.2e})")
    print(f"slope (real ~ predicted, linear fit) = {slope:.4f}  (1.0 = perfectly proportional)")
    print(f"same-sign rate: {100.0 * same_sign.float().mean():.1f}%")
    print(f"v6 reduces |error| vs NNUE alone: {100.0 * improved.float().mean():.1f}% of positions")
    print(f"mean |error|  NNUE alone: {nnue_abs_error.mean():.4g}   NNUE+v6: {v6_abs_error.mean():.4g}   "
          f"constant-shift baseline: {constant_abs_error.mean():.4g}")
    print()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    binpack = os.path.join(here, "..", "..", "datasets", "test80-2024-01-jan-2tb7p.min-v2.v6.binpack")
    nnue_path = os.path.join(here, "..", "..", "..", "data", "nnue", "v3.nnue")
    v6_ckpt = os.path.join(here, "..", "checkpoints_v6", "chesscnn_final.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    dataset = PlaneBatchDataset(
        filenames=[binpack], batch_size=NUM_POSITIONS, cyclic=False,
        num_workers=1, val_percent=5, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, result, piece_count, nnue_score = next(iter(dataset))
    score, result, nnue_score = score.squeeze(-1), result.squeeze(-1), nnue_score.squeeze(-1)

    mask = score.abs() < MATE_ABS_THRESHOLD
    planes, score, result, piece_count, nnue_score = (
        planes[mask], score[mask], result[mask], piece_count[mask], nnue_score[mask]
    )
    planes, score, result, piece_count, nnue_score = (
        planes.to(device), score.to(device), result.to(device), piece_count.to(device), nnue_score.to(device)
    )
    print(f"sample: {planes.shape[0]} positions, device={device}\n")

    calibration = NnueCalibration(os.path.join(here, "nnue_calibration.json")).to(device)
    nnue_logit = calibration(nnue_score)
    nnue_cp = nnue_logit * SCORE_SCALE
    nnue_wdl = torch.tanh(nnue_logit / SCORE_SCALE)

    v6_model, is_v5_v6, _ = load_model(v6_ckpt, device)
    assert is_v5_v6
    v6_planes = planes[:, :v6_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v6_logit = v6_model(v6_planes, piece_count, nnue_logit)
    v6_cp = v6_logit * SCORE_SCALE
    v6_wdl = torch.tanh(v6_logit / SCORE_SCALE)

    analyze("cp space (vs Stockfish searched score)", score, nnue_cp, v6_cp)
    analyze("WDL space (vs real game outcome, tanh(logit/SCORE_SCALE))", result, nnue_wdl, v6_wdl)


if __name__ == "__main__":
    main()
