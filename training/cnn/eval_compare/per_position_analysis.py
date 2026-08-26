"""Per-position analysis: does the CNN ever beat NNUE, or is it just
uniformly weaker?

Correction (async GPU eval blended with NNUE) only makes sense if the
CNN's errors are DIFFERENT from NNUE's, not just larger on average —
two models with correlated errors add nothing when combined; only
models that are wrong in different places can correct each other.

This compares |NNUE - stockfish_static| vs |CNN - stockfish_static|
position by position, using the cached ground truth from
compare_checkpoints.py, and reports:
  - how many positions the CNN is actually closer on ("CNN wins")
  - the correlation between the two error series (high = redundant
    errors, no complementary signal; low = genuinely different
    blind spots, a real case for correction)
  - piece-count profile of the positions where CNN wins, to see if
    there's an identifiable pattern (phase-dependent, say)
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_checkpoints import (  # noqa: E402
    DEFAULT_CACHE, MATE_SENTINEL_CP, fen_to_planes_and_piece_count,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import ChessCNN  # noqa: E402


def correlation(a, b):
    a_t, b_t = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    a_c, b_c = a_t - a_t.mean(), b_t - b_t.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.cache) as f:
        gt = json.load(f)
    fens = gt["fens"]
    sf, sf_static, nnue = gt["stockfish_searched"], gt["stockfish_static"], gt["nnue"]

    keep_idx = [i for i, (s, static) in enumerate(zip(sf, sf_static))
                if abs(s) < MATE_SENTINEL_CP and static is not None]
    print(f"{len(keep_idx)}/{len(fens)} positions kept (mate/in-check excluded)\n")

    device = torch.device("cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ChessCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"CNN checkpoint step: {ckpt['step']}\n")

    planes_list, pc_list = [], []
    for i in keep_idx:
        planes, pc = fen_to_planes_and_piece_count(fens[i])
        planes_list.append(planes)
        pc_list.append(pc)
    planes_batch = torch.stack(planes_list)
    pc_batch = torch.tensor(pc_list, dtype=torch.int64)

    with torch.no_grad():
        cnn_logit = model(planes_batch, pc_batch)  # raw logit, before any cp_scale
        cnn_pred_uncalibrated = (410.0 * cnn_logit).tolist()

    sf_static_f = [sf_static[i] for i in keep_idx]
    logit_t = cnn_logit.squeeze(-1) if cnn_logit.dim() > 1 else cnn_logit
    sf_t = torch.tensor(sf_static_f, dtype=torch.float32)

    # Two-fold split of WAC itself: fit the scale on one half, evaluate
    # the win-rate on the OTHER half — no position is ever scored with
    # a scale fit on itself. A binpack-based calibration was tried and
    # rejected: the binpack's score field is a SEARCHED Stockfish score
    # (what the CNN was trained to approximate via the lambda blend),
    # not the STATIC eval used as ground truth here — different
    # quantities, so a scale fit on one doesn't transfer to the other
    # (measured 608.8 vs ~91, a ~7x mismatch). Splitting WAC keeps the
    # target consistent (static eval throughout) while still avoiding
    # any circularity.
    n = len(keep_idx)
    fold_a = list(range(0, n, 2))
    fold_b = list(range(1, n, 2))

    def fit_scale(idx):
        return (logit_t[idx] * sf_t[idx]).sum() / (logit_t[idx] * logit_t[idx]).sum()

    scale_from_a = fit_scale(fold_a)
    scale_from_b = fit_scale(fold_b)
    print(f"Scale fit on fold A (n={len(fold_a)}): {scale_from_a.item():.1f}")
    print(f"Scale fit on fold B (n={len(fold_b)}): {scale_from_b.item():.1f}")

    # Each fold is scored with the scale fit on the OTHER fold.
    cnn_pred = [None] * n
    for i in fold_a:
        cnn_pred[i] = (scale_from_b * logit_t[i]).item()
    for i in fold_b:
        cnn_pred[i] = (scale_from_a * logit_t[i]).item()
    print("(each fold scored with the other fold's scale — cross-validated, "
          "no self-fit contamination)\n")
    nnue_f = [nnue[i] for i in keep_idx]
    fens_f = [fens[i] for i in keep_idx]
    pc_f = pc_list

    nnue_err = [abs(n - s) for n, s in zip(nnue_f, sf_static_f)]
    cnn_err = [abs(c - s) for c, s in zip(cnn_pred, sf_static_f)]
    cnn_err_uncalibrated = [abs(c - s) for c, s in zip(cnn_pred_uncalibrated, sf_static_f)]

    cnn_wins_uncalibrated = sum(1 for i in range(len(keep_idx)) if cnn_err_uncalibrated[i] < nnue_err[i])
    cnn_wins = [i for i in range(len(keep_idx)) if cnn_err[i] < nnue_err[i]]
    print(
        f"CNN win rate at SCORE_SCALE=410 (uncalibrated): "
        f"{cnn_wins_uncalibrated}/{len(keep_idx)} ({100 * cnn_wins_uncalibrated / len(keep_idx):.1f}%)"
    )
    print(
        f"CNN win rate at refit optimal scale:             "
        f"{len(cnn_wins)}/{len(keep_idx)} ({100 * len(cnn_wins) / len(keep_idx):.1f}%)\n"
    )

    err_corr = correlation(nnue_err, cnn_err)
    print(f"Correlation between NNUE's and CNN's error magnitude: {err_corr:.4f}")
    print("(high = both models tend to be wrong on the same positions — redundant, "
          "no complementary signal. low/negative = genuinely different blind spots.)\n")

    if cnn_wins:
        win_pcs = [pc_f[i] for i in cnn_wins]
        all_pcs = pc_f
        print(f"Piece count where CNN wins: mean={sum(win_pcs)/len(win_pcs):.1f}, "
              f"vs overall mean={sum(all_pcs)/len(all_pcs):.1f}")

        print("\nTop 10 positions where CNN wins by the largest margin:")
        margins = sorted(cnn_wins, key=lambda i: nnue_err[i] - cnn_err[i], reverse=True)[:10]
        for i in margins:
            print(
                f"  fen={fens_f[i]}  piece_count={pc_f[i]:2d}  "
                f"sf_static={sf_static_f[i]:+5d}  nnue={nnue_f[i]:+5d} (err {nnue_err[i]:.0f})  "
                f"cnn={cnn_pred[i]:+.0f} (err {cnn_err[i]:.0f})"
            )
    else:
        print("CNN did not beat NNUE on a single position in this test set.")


if __name__ == "__main__":
    main()
