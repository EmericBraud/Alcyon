"""Applies a monotonic piecewise-linear NNUE calibration curve (see
eval_compare/calibrate_nnue_scale.py) to convert a raw NNUE centipawn
score into a logit, on the same scale as the rest of the lambda-blended
target (see train.py's SCORE_SCALE).

A single scalar scale is not enough here -- see
calibrate_nnue_scale.py's module docstring for why the raw-NNUE-to-
target relationship is measurably non-linear (magnitude-dependent),
and why the CNN trunk can't correct that itself (it never sees
nnue_score as an input).
"""

import json

import torch


class NnueCalibration:
    def __init__(self, path: str):
        with open(path) as f:
            data = json.load(f)
        # Breakpoints are built from sorted quantile bins (see
        # calibrate_nnue_scale.py), so x is already ascending -- required
        # by torch.searchsorted below.
        self.x = torch.tensor(data["x"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def to(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        return self

    def __call__(self, nnue_score: torch.Tensor) -> torch.Tensor:
        """nnue_score: (batch,) raw NNUE centipawn scores. Returns the
        calibrated logit, same shape. Values outside the fitted range
        extrapolate linearly using the nearest end segment's slope
        (idx is clamped, but the interpolation fraction `t` is not, so
        it can go outside [0, 1] for out-of-range inputs)."""
        idx = torch.searchsorted(self.x, nnue_score)
        idx = idx.clamp(1, self.x.shape[0] - 1)
        x0, x1 = self.x[idx - 1], self.x[idx]
        y0, y1 = self.y[idx - 1], self.y[idx]
        t = (nnue_score - x0) / (x1 - x0 + 1e-9)
        return y0 + t * (y1 - y0)
