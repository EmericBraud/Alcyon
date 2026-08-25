"""CNN position evaluator, complementary to the engine's NNUE.

Value-only network (no policy head — move search stays in the C++
alpha-beta engine). Output is WDL (win/draw/loss), converted to
centipawns at inference time using a calibrated sigmoid, to stay on
the same score scale as the NNUE eval.

Input: dense 8x8xN board planes (see PLANES below), one stack per
side to move already oriented so the side to move is always "us".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Input plane layout -----------------------------------------------
# Order matters: the C++ data loader (training/cnn/data_loader) must
# fill planes in exactly this order.
#
#   0-5   : our pieces   (pawn, knight, bishop, rook, queen, king)
#   6-11  : their pieces (pawn, knight, bishop, rook, queen, king)
#   12    : side to move is white (1.0 / 0.0, constant plane)
#   13-16 : castling rights (us kingside, us queenside,
#                             them kingside, them queenside)
#   17    : en passant target square (one-hot, all zeros if none)
#   18    : halfmove clock (rule50 counter), normalized to [0, 1]
#
# Repetition is intentionally NOT included — the binpack format has
# no game-history window to derive it from, and the search already
# handles repetition detection on its own. See
# docs/gpu-async-eval/architecture.md.
NUM_PLANES = 19
BOARD_SIZE = 8


class ResidualBlock(nn.Module):
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


class ChessCNN(nn.Module):
    """8 residual blocks x 96 channels, WDL value head."""

    def __init__(self, channels=96, num_blocks=8, num_planes=NUM_PLANES):
        super().__init__()
        self.stem_conv = nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(num_blocks))

        # Value head: global average pool -> small MLP -> 3-way WDL logits.
        self.value_fc1 = nn.Linear(channels, channels)
        self.value_fc2 = nn.Linear(channels, 3)  # [win, draw, loss] logits

    def forward(self, planes):
        # planes: (batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
        x = F.relu(self.stem_bn(self.stem_conv(planes)))
        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=(2, 3))  # global average pool -> (batch, channels)
        h = F.relu(self.value_fc1(pooled))
        wdl_logits = self.value_fc2(h)
        return wdl_logits

    @torch.no_grad()
    def eval_centipawns(self, planes, cp_scale=410.0):
        """WDL logits -> expected score in [0, 1] -> centipawns.

        cp_scale must be calibrated against the engine's NNUE score
        scale on a held-out set (see scripts/calibrate_cp_scale.py,
        not yet written) before this is trustworthy for the async
        GPU eval cache.
        """
        wdl = F.softmax(self.forward(planes), dim=-1)
        win, draw, loss = wdl[:, 0], wdl[:, 1], wdl[:, 2]
        expected_score = win + 0.5 * draw  # in [0, 1]
        # Inverse logistic, same family Stockfish uses to convert
        # win probability to a centipawn-like score.
        expected_score = expected_score.clamp(1e-6, 1 - 1e-6)
        return cp_scale * torch.log(expected_score / (1 - expected_score))
