"""Which is the better predictor of "NNUE is wrong here": v3's raw
disagreement with NNUE, or v6's implied correction magnitude?

Both are candidate signals for the engine's consultative GPU-eval design
(transp_table.hpp's classify_cut_decision): a large disagreement between
NNUE and a second model is only useful if it actually predicts that NNUE
is likely to be wrong. This compares the two candidate second models
head-to-head on that exact question, in both cp space (scale-sensitive,
see disagreement_correlation.py) and WDL/probability space (the metric
that actually matters -- see v6_residual_proportionality.py's finding
that cp-space comparisons can be misleading here).

v6 checkpoint: chesscnn_step200000.pt (the "extend-steps" run's final
checkpoint step -- chesscnn_final.pt on disk currently resolves to an
EARLIER step=100000 snapshot, not the fully-trained 200k run, see
results_log.csv's last v6 row).

Same held-out test split / normalization as the other eval_compare
scripts in this session (calibrated NNUE logit + SCORE_SCALE=410).
"""

import os
import sys

import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from data_loader import PlaneBatchDataset  # noqa: E402
from nnue_calibration import NnueCalibration  # noqa: E402
from model_loader import load_model  # noqa: E402

SCORE_SCALE = 410.0
MATE_ABS_THRESHOLD = 8000
NUM_POSITIONS = 65536
TOP_FRACTION = 0.20  # "NNUE is wrong" := in the top 20% worst |error| positions


def correlation(a, b):
    a_c, b_c = a - a.mean(), b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def evaluate_predictor(name, disagreement, nnue_error):
    disagreement_np = disagreement.detach().cpu().numpy()
    nnue_error_np = nnue_error.detach().cpu().numpy()

    pearson = correlation(disagreement, nnue_error)
    spearman, _ = spearmanr(disagreement_np, nnue_error_np)
    kendall, _ = kendalltau(disagreement_np, nnue_error_np)

    threshold = torch.quantile(nnue_error, 1.0 - TOP_FRACTION)
    is_bad = (nnue_error >= threshold).detach().cpu().numpy().astype(int)
    auc = roc_auc_score(is_bad, disagreement_np)

    print(f"  {name:28s}  Pearson={pearson:.4f}  Spearman={spearman:.4f}  "
          f"Kendall={kendall:.4f}  AUC(top {int(TOP_FRACTION*100)}% worst)={auc:.4f}")
    return auc


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    binpack = os.path.join(here, "..", "..", "datasets", "test80-2024-01-jan-2tb7p.min-v2.v6.binpack")
    nnue_path = os.path.join(here, "..", "..", "..", "data", "nnue", "v3.nnue")
    v3_ckpt = os.path.join(here, "..", "checkpoints", "chesscnn_v3_step300000.pt")
    v6_ckpt = os.path.join(here, "..", "checkpoints_v6", "chesscnn_step200000.pt")

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

    v3_model, is_v5_v3, v3_ckpt_data = load_model(v3_ckpt, device)
    assert not is_v5_v3
    v3_planes = planes[:, :v3_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v3_logit = v3_model(v3_planes, piece_count)
    v3_cp = v3_logit * SCORE_SCALE
    v3_wdl = torch.tanh(v3_logit / SCORE_SCALE)

    v6_model, is_v5_v6, v6_ckpt_data = load_model(v6_ckpt, device)
    assert is_v5_v6
    v6_planes = planes[:, :v6_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v6_logit = v6_model(v6_planes, piece_count, nnue_logit)
    v6_cp = v6_logit * SCORE_SCALE
    v6_wdl = torch.tanh(v6_logit / SCORE_SCALE)

    print(f"v3 checkpoint step: {v3_ckpt_data['step']}   v6 checkpoint step: {v6_ckpt_data['step']}\n")

    # "NNUE is wrong" -- the thing we're trying to predict, in each space.
    nnue_error_cp = (nnue_cp - score).abs()
    nnue_error_wdl = (nnue_wdl - result).abs()

    disagreement_v3_cp = (nnue_cp - v3_cp).abs()
    disagreement_v6_cp = (nnue_cp - v6_cp).abs()
    disagreement_v3_wdl = (nnue_wdl - v3_wdl).abs()
    disagreement_v6_wdl = (nnue_wdl - v6_wdl).abs()

    print("=== cp space: predicting |nnue_cp - stockfish_score| ===")
    auc_v3_cp = evaluate_predictor("v3 disagreement (|nnue-v3|)", disagreement_v3_cp, nnue_error_cp)
    auc_v6_cp = evaluate_predictor("v6 disagreement (|nnue-v6|)", disagreement_v6_cp, nnue_error_cp)

    print("\n=== WDL space: predicting |nnue_wdl - real_result| ===")
    auc_v3_wdl = evaluate_predictor("v3 disagreement (|nnue-v3|)", disagreement_v3_wdl, nnue_error_wdl)
    auc_v6_wdl = evaluate_predictor("v6 disagreement (|nnue-v6|)", disagreement_v6_wdl, nnue_error_wdl)

    print("\n=== Verdict ===")
    print(f"cp space:  {'v3' if auc_v3_cp > auc_v6_cp else 'v6'} wins "
          f"(AUC {max(auc_v3_cp, auc_v6_cp):.4f} vs {min(auc_v3_cp, auc_v6_cp):.4f})")
    print(f"WDL space: {'v3' if auc_v3_wdl > auc_v6_wdl else 'v6'} wins "
          f"(AUC {max(auc_v3_wdl, auc_v6_wdl):.4f} vs {min(auc_v3_wdl, auc_v6_wdl):.4f})")


if __name__ == "__main__":
    main()
