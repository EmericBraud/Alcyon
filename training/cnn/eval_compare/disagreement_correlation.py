"""Is a NNUE/CNN disagreement (on the engine's actual cutoff-classification
sense) correlated with NNUE actually being wrong vs. ground truth?

This is the empirical check behind the engine's "consultative" GPU-eval
design (see transp_table.hpp's classify_cut_decision): we only trust the
GPU (CNN) score to override the main TT when NNUE and the CNN agree on
whether a position would fail low / stay neutral / fail high. The
assumption behind that design is that DISAGREEMENT is itself informative
-- it should correlate with NNUE actually being wrong, which is what
would justify falling back to a real search there instead of trusting
either score. This script tests that assumption directly on held-out
data rather than assuming it.

Ground truth: the binpack's own Stockfish-searched score (continuous,
less noisy than WDL) -- same methodology as error_correlation_v3_v6.py
and experiment-v1.md's per_position_analysis.py. Same held-out test
split (is_validation=True, val_percent=5) as all this project's other
eval_compare scripts, so this sample was never seen during v3's training.
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

# Proxy for the engine's real alpha-beta window (which varies node to
# node) -- a fixed symmetric band around 0 centipawns. Mirrors
# classify_cut_decision's Low/Neutral/High three-way split.
MARGIN_CP = 50.0


def classify(cp: torch.Tensor, margin: float) -> torch.Tensor:
    # 0 = Low (<=-margin), 1 = Neutral, 2 = High (>=+margin)
    return torch.where(cp <= -margin, 0, torch.where(cp >= margin, 2, 1))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    binpack = os.path.join(here, "..", "..", "datasets", "test80-2024-01-jan-2tb7p.min-v2.v6.binpack")
    nnue_path = os.path.join(here, "..", "..", "..", "data", "nnue", "v3.nnue")
    v3_ckpt = os.path.join(here, "..", "checkpoints", "chesscnn_v3_step300000.pt")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # num_workers=1: cross-run sampling noise otherwise dominates the
    # signal at this sample size (see eval_compare_sampling_noise memory).
    dataset = PlaneBatchDataset(
        filenames=[binpack], batch_size=NUM_POSITIONS, cyclic=False,
        num_workers=1, val_percent=5, is_validation=True,
        nnue_path=nnue_path,
    )
    planes, score, _result, piece_count, nnue_score = next(iter(dataset))
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

    v3_model, is_v5_v3, _ = load_model(v3_ckpt, device)
    assert not is_v5_v3
    v3_planes = planes[:, :v3_model.stem_conv.weight.shape[1]]
    with torch.no_grad():
        v3_logit = v3_model(v3_planes, piece_count)
    v3_cp = v3_logit * SCORE_SCALE

    # Ground truth is the binpack's own Stockfish score, same cp scale
    # (both nnue_cp and v3_cp were calibrated/scaled to match it above --
    # this is the exact normalization step error_correlation_v3_v6.py
    # already relies on: nnue_error = nnue_cp - score there too).
    nnue_abs_error = (nnue_cp - score).abs()

    for margin in (10.0, 25.0, 50.0, 100.0):
        nnue_class = classify(nnue_cp, margin)
        v3_class = classify(v3_cp, margin)
        disagree = nnue_class != v3_class

        n_agree, n_disagree = (~disagree).sum().item(), disagree.sum().item()
        print(f"--- margin = {margin:.0f}cp ---")
        print(f"agree:    {n_agree:6d} positions ({100.0 * n_agree / len(disagree):.1f}%)")
        print(f"disagree: {n_disagree:6d} positions ({100.0 * n_disagree / len(disagree):.1f}%)")
        print("NNUE |error| vs Stockfish ground truth:")
        print(f"  when agree:    mean={nnue_abs_error[~disagree].mean():.1f}  median={nnue_abs_error[~disagree].median():.1f}")
        print(f"  when disagree: mean={nnue_abs_error[disagree].mean():.1f}  median={nnue_abs_error[disagree].median():.1f}\n")

    # Use MARGIN_CP for the headline agree/disagree split below.
    nnue_class = classify(nnue_cp, MARGIN_CP)
    v3_class = classify(v3_cp, MARGIN_CP)
    disagree = nnue_class != v3_class

    # Continuous version of the same question: does the RAW disagreement
    # magnitude (|nnue_cp - v3_cp|, not just the 3-way classification)
    # correlate with how wrong NNUE actually is?
    disagreement_magnitude = (nnue_cp - v3_cp).abs()
    a, b = disagreement_magnitude - disagreement_magnitude.mean(), nnue_abs_error - nnue_abs_error.mean()
    corr = ((a * b).sum() / (a.norm() * b.norm() + 1e-9)).item()
    print(f"corr(|nnue_cp - v3_cp|, |nnue_cp - stockfish|) = {corr:.4f}\n")

    mean_ratio = (nnue_abs_error[disagree].mean() / nnue_abs_error[~disagree].mean()).item()
    median_ratio = (nnue_abs_error[disagree].median() / nnue_abs_error[~disagree].median()).item()
    print(f"mean |error| ratio (disagree/agree):   {mean_ratio:.2f}x")
    print(f"median |error| ratio (disagree/agree): {median_ratio:.2f}x")
    print("(mean can be dominated by a few heavy-tailed outliers on either "
          "side; median is the more robust comparison here.)\n")

    if median_ratio > 1.0 and corr > 0.0:
        print("Disagreement IS correlated with NNUE being more wrong -- "
              "supports the consultative design's core assumption.")
    else:
        print("Disagreement is NOT clearly correlated with NNUE being more "
              "wrong on this sample -- the consultative design's core "
              "assumption is not well supported here.")


if __name__ == "__main__":
    main()
