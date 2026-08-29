"""Loads a CNN checkpoint of ANY version (v1/v3/v4 legacy, or v5) for
analysis, auto-detecting the architecture directly from the state_dict
rather than requiring the caller to know/guess which version produced
a given checkpoint file.

v5 checkpoints (no "psqt.conv.weight" key -- the PSQT skip was
dropped, see model.py) use the current model.py's ChessCNN() with its
fixed defaults (v5 was only ever trained at one size). Legacy
checkpoints (v1/v3/v4) use legacy_model.py's frozen copy of the pre-v5
architecture, with channels/num_blocks/num_planes/num_phase_buckets
inferred from the state_dict's own tensor shapes, and the SE-gate
presence (absent only in v1) inferred from whether any
"blocks.0.se.*" key exists.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import legacy_model  # noqa: E402
from model import ChessCNN  # noqa: E402

_original_residual_block = legacy_model.ResidualBlock


def _legacy_residual_block_no_se():
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlockNoSE(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return F.relu(out + residual)

    return ResidualBlockNoSE


def load_model(ckpt_path, device):
    """Returns (model, is_v5, ckpt). `model` is already .eval()'d and on
    `device`. `ckpt` is the raw loaded dict (has "step" and, for v5,
    "nnue_path"/"nnue_calibration" -- see train.py)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt["model"]
    is_v5 = "psqt.conv.weight" not in state_dict

    if is_v5:
        model = ChessCNN()
    else:
        channels = state_dict["stem_conv.weight"].shape[0]
        num_planes = state_dict["stem_conv.weight"].shape[1]
        num_blocks = len({k.split(".")[1] for k in state_dict if k.startswith("blocks.")})
        num_phase_buckets = state_dict["psqt.conv.weight"].shape[0]
        has_se = any(k.startswith("blocks.0.se.") for k in state_dict)

        # Reset before every load: a previous legacy load in the same
        # process may have monkeypatched this to the no-SE (v1) variant.
        legacy_model.ResidualBlock = _original_residual_block if has_se else _legacy_residual_block_no_se()

        # v1/v3 used 4 buckets (24, 16, 8); only v4 switched to the
        # 8-bucket (26, 22, 18, 14, 10, 6, 3) scheme -- legacy_model.py's
        # module-level PHASE_BUCKET_BOUNDARIES defaults to the v4 one, so
        # patch it back for 4-bucket checkpoints. phase_bucket_for_piece_count
        # reads this module global at call time, so patching the attribute
        # (not the function) is enough.
        if num_phase_buckets == 4:
            legacy_model.PHASE_BUCKET_BOUNDARIES = (24, 16, 8)
        elif num_phase_buckets == 8:
            legacy_model.PHASE_BUCKET_BOUNDARIES = (26, 22, 18, 14, 10, 6, 3)
        else:
            raise ValueError(f"unexpected num_phase_buckets={num_phase_buckets} in {ckpt_path}")

        model = legacy_model.ChessCNN(
            channels=channels, num_blocks=num_blocks,
            num_planes=num_planes, num_phase_buckets=num_phase_buckets,
        )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, is_v5, ckpt
