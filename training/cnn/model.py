"""CNN residual-error corrector for the engine's NNUE (v5).

Value-only network (no policy head — move search stays in the C++
alpha-beta engine). Predicts a correction on top of a precomputed,
frozen NNUE score: `logit_final = trunk_logits + nnue_logit`, where
`nnue_logit` is supplied by the data loader (chess26's own NNUE,
evaluated once per position, treated as a non-trainable constant —
no gradient flows into it). Trained against the same lambda-blended
target of the Stockfish search score and the actual game result as
before (see train.py's lambda annealing schedule and
docs/gpu-async-eval/v5-hybrid-nnue-cnn.md for the full rationale) —
the trunk implicitly learns `target - nnue_logit`, i.e. NNUE's
residual error, without needing to reformulate the loss.

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
#   19-24 : squares attacked by our pawn/knight/bishop/rook/queen/king
#           (binary, pseudo-legal, one plane per piece type)
#   25-30 : same, for their pieces
#   31    : Chebyshev distance to our king, normalized to [0, 1]
#   32    : Chebyshev distance to their king, normalized to [0, 1]
#
# Repetition is intentionally NOT included — the binpack format has
# no game-history window to derive it from, and the search already
# handles repetition detection on its own. See
# docs/gpu-async-eval/architecture.md.
NUM_PLANES = 33
BOARD_SIZE = 8

# --- Phase buckets ------------------------------------------------------
# Non-king piece count (both sides, see plane_batch.h's piece_count
# field) determines which value head evaluates a given position.
# 8 buckets, ~4 pieces per bucket -- matches the granularity of the
# NNUE's own 8 layer-stack/PSQT buckets (bucket_for_piece_count in
# nnue_model.hpp, which spaces its 8 buckets every 4 pieces out of a
# 32-piece range that includes kings; ours excludes kings, hence the
# range 0-30 instead of 2-32).
#
#   bucket 0: >= 26 pieces
#   bucket 1: 22-25 pieces
#   bucket 2: 18-21 pieces
#   bucket 3: 14-17 pieces
#   bucket 4: 10-13 pieces
#   bucket 5: 6-9 pieces
#   bucket 6: 3-5 pieces
#   bucket 7: < 3 pieces    (near-empty endgame)
PHASE_BUCKET_BOUNDARIES = (26, 22, 18, 14, 10, 6, 3)
NUM_PHASE_BUCKETS = len(PHASE_BUCKET_BOUNDARIES) + 1


def phase_bucket_for_piece_count(piece_count: torch.Tensor) -> torch.Tensor:
    """Maps a (batch,) int64 piece_count tensor to bucket indices in [0, NUM_PHASE_BUCKETS)."""
    bucket = torch.zeros_like(piece_count)
    for boundary in PHASE_BUCKET_BOUNDARIES:
        bucket += (piece_count < boundary).to(bucket.dtype)
    return bucket


class SqueezeExcitation(nn.Module):
    """Channel attention: pools each channel to a scalar, learns a
    per-channel gate from that summary, and rescales the channel by
    it — lets the trunk dynamically emphasize/suppress whole feature
    channels per position (e.g. "king-safety channels matter a lot
    here") instead of treating every channel uniformly. Same block
    used in Leela Chess Zero's trunk; cheap (a 2-layer MLP over a
    pooled vector) relative to the convolutions around it.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced)
        self.fc2 = nn.Linear(reduced, channels)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcitation(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual)


class PhaseValueHead(nn.Module):
    """One small value head — MLP -> scalar win-prob logit.

    Takes the concatenation of mid-trunk and final pooled features
    (see ChessCNN.forward), not just the final ones — analogous to
    NNUE's output layer receiving both L1 and L2 activations directly
    (DenseLayer<2*L2+2*L3, 1> in nnue_model.hpp), rather than only the
    deepest layer's output. Gives the head a shorter path to less-
    transformed features alongside the fully-processed ones.
    """

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels * 2, channels)
        self.fc2 = nn.Linear(channels, 1)

    def forward(self, pooled_mid, pooled_final):
        h = F.relu(self.fc1(torch.cat([pooled_mid, pooled_final], dim=-1)))
        return self.fc2(h).squeeze(-1)  # (batch,)


class ChessCNN(nn.Module):
    """8 residual blocks x 96 channels (each with a squeeze-excitation
    channel-attention gate, see SqueezeExcitation), shared trunk, one
    scalar head per phase bucket. No PSQT skip: the frozen `nnue_logit`
    term (see forward()) already carries the NNUE's own PSQT signal,
    so a separate linear material skip here would be redundant (see
    docs/gpu-async-eval/v5-hybrid-nnue-cnn.md).

    Deliberately kept at v1's size, not v4's (20x224) — the trunk's
    job here is to predict a residual correction, an easier task than
    being a standalone evaluator, so it doesn't need v4's capacity.
    Revisit (e.g. reduce further) once ablations on the residual task
    itself are available.

    The trunk is trained on every position regardless of phase, so it
    keeps the statistical benefit of the full dataset; only the small
    per-bucket heads specialize (see PHASE_BUCKET_BOUNDARIES above).
    """

    def __init__(self, channels=96, num_blocks=8, num_planes=NUM_PLANES,
                 num_phase_buckets=NUM_PHASE_BUCKETS):
        super().__init__()
        self.stem_conv = nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(num_blocks))
        self.heads = nn.ModuleList(PhaseValueHead(channels) for _ in range(num_phase_buckets))

    def forward(self, planes, piece_count, nnue_logit):
        # planes: (batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
        # piece_count: (batch,) int64, non-king piece count from the loader.
        # nnue_logit: (batch,) float, precomputed NNUE score converted to a
        #   win-probability logit — a non-trainable constant, not a model
        #   parameter (no gradient flows into it; the loader supplies it
        #   already detached).
        # returns: (batch,) win-probability logits (pass through sigmoid for [0, 1]).
        bucket = phase_bucket_for_piece_count(piece_count)

        x = F.relu(self.stem_bn(self.stem_conv(planes)))
        mid_point = len(self.blocks) // 2
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == mid_point - 1:
                pooled_mid = x.mean(dim=(2, 3))  # (batch, channels), less-transformed

        pooled_final = x.mean(dim=(2, 3))  # (batch, channels), fully-transformed

        trunk_logits = torch.empty(planes.shape[0], device=planes.device, dtype=pooled_final.dtype)
        for bucket_idx, head in enumerate(self.heads):
            mask = bucket == bucket_idx
            if mask.any():
                trunk_logits[mask] = head(pooled_mid[mask], pooled_final[mask])

        return trunk_logits + nnue_logit

    @torch.no_grad()
    def eval_centipawns(self, planes, piece_count, nnue_logit, cp_scale=410.0):
        """Win-probability logit -> centipawns, same sigmoid family Stockfish
        uses to convert win probability to a centipawn-like score.

        cp_scale must be calibrated against the engine's NNUE score
        scale on a held-out set (see scripts/calibrate_cp_scale.py,
        not yet written) before this is trustworthy for the async
        GPU eval cache. A single global cp_scale is used across all
        phase buckets for now — revisit if calibration turns out to
        need per-bucket scales too.
        """
        logit = self.forward(planes, piece_count, nnue_logit)
        # logit is already logit(p); centipawns = cp_scale * logit(p) is the
        # same transform as cp_scale * log(p / (1 - p)) applied to sigmoid(logit).
        return cp_scale * logit
