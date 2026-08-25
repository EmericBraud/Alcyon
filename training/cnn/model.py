"""CNN position evaluator, complementary to the engine's NNUE.

Value-only network (no policy head — move search stays in the C++
alpha-beta engine). Output is a single scalar win-probability logit,
trained against a lambda-blended target of the Stockfish search score
and the actual game result (see train.py's lambda annealing schedule
and docs/gpu-async-eval/ for the rationale) — not a 3-way WDL
classification, since the blended target is a continuous probability,
not a discrete class.

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
NUM_PIECE_PLANES = 12  # planes 0-11: the piece-placement planes only

# --- Phase buckets ------------------------------------------------------
# Non-king piece count (both sides, see plane_batch.h's piece_count
# field) determines which value head evaluates a given position.
# Boundaries are a starting point, not tuned — revisit once real
# training data shows how positions distribute across buckets.
#
#   bucket 0: >= 24 pieces  (opening / early middlegame)
#   bucket 1: 16-23 pieces  (middlegame)
#   bucket 2: 8-15 pieces   (late middlegame / early endgame)
#   bucket 3: < 8 pieces    (endgame)
PHASE_BUCKET_BOUNDARIES = (24, 16, 8)
NUM_PHASE_BUCKETS = len(PHASE_BUCKET_BOUNDARIES) + 1


def phase_bucket_for_piece_count(piece_count: torch.Tensor) -> torch.Tensor:
    """Maps a (batch,) int64 piece_count tensor to bucket indices in [0, NUM_PHASE_BUCKETS)."""
    bucket = torch.zeros_like(piece_count)
    for boundary in PHASE_BUCKET_BOUNDARIES:
        bucket += (piece_count < boundary).to(bucket.dtype)
    return bucket


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


class PhaseValueHead(nn.Module):
    """One small value head — global average pool -> MLP -> scalar win-prob logit."""

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, 1)

    def forward(self, pooled):
        h = F.relu(self.fc1(pooled))
        return self.fc2(h).squeeze(-1)  # (batch,)


class PSQTHead(nn.Module):
    """Direct linear piece-square skip, analogous to NNUE's PSQT accumulator.

    Bypasses the residual trunk entirely: one learnable per-square,
    per-piece-plane weight, summed over the board, with its own value
    per phase bucket (mirrors NNUE's separate PSQT weights per
    bucket). Added directly to the trunk's logit output — gives the
    network a stable linear material/positional baseline to correct,
    instead of having to rediscover material counting indirectly
    through convolutions and nonlinear heads.
    """

    def __init__(self, num_piece_planes=NUM_PIECE_PLANES, num_phase_buckets=NUM_PHASE_BUCKETS):
        super().__init__()
        self.conv = nn.Conv2d(num_piece_planes, num_phase_buckets, kernel_size=1, bias=False)

    def forward(self, piece_planes, bucket):
        # piece_planes: (batch, NUM_PIECE_PLANES, BOARD_SIZE, BOARD_SIZE)
        # bucket: (batch,) int64 phase bucket index.
        per_bucket = self.conv(piece_planes).sum(dim=(2, 3))  # (batch, num_phase_buckets)
        return per_bucket.gather(1, bucket.unsqueeze(1)).squeeze(1)  # (batch,)


class ChessCNN(nn.Module):
    """8 residual blocks x 96 channels, shared trunk, one scalar head per
    phase bucket, plus a direct linear PSQT skip (see PSQTHead).

    The trunk is trained on every position regardless of phase, so it
    keeps the statistical benefit of the full dataset; only the small
    per-bucket heads and PSQT weights specialize (see
    PHASE_BUCKET_BOUNDARIES above).
    """

    def __init__(self, channels=96, num_blocks=8, num_planes=NUM_PLANES,
                 num_phase_buckets=NUM_PHASE_BUCKETS):
        super().__init__()
        self.stem_conv = nn.Conv2d(num_planes, channels, kernel_size=3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(num_blocks))
        self.heads = nn.ModuleList(PhaseValueHead(channels) for _ in range(num_phase_buckets))
        self.psqt = PSQTHead(num_phase_buckets=num_phase_buckets)

    def forward(self, planes, piece_count):
        # planes: (batch, NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
        # piece_count: (batch,) int64, non-king piece count from the loader.
        # returns: (batch,) win-probability logits (pass through sigmoid for [0, 1]).
        bucket = phase_bucket_for_piece_count(piece_count)

        x = F.relu(self.stem_bn(self.stem_conv(planes)))
        for block in self.blocks:
            x = block(x)

        pooled = x.mean(dim=(2, 3))  # global average pool -> (batch, channels)

        trunk_logits = torch.empty(planes.shape[0], device=planes.device, dtype=pooled.dtype)
        for bucket_idx, head in enumerate(self.heads):
            mask = bucket == bucket_idx
            if mask.any():
                trunk_logits[mask] = head(pooled[mask])

        psqt_logits = self.psqt(planes[:, :NUM_PIECE_PLANES], bucket)
        return trunk_logits + psqt_logits

    @torch.no_grad()
    def eval_centipawns(self, planes, piece_count, cp_scale=410.0):
        """Win-probability logit -> centipawns, same sigmoid family Stockfish
        uses to convert win probability to a centipawn-like score.

        cp_scale must be calibrated against the engine's NNUE score
        scale on a held-out set (see scripts/calibrate_cp_scale.py,
        not yet written) before this is trustworthy for the async
        GPU eval cache. A single global cp_scale is used across all
        phase buckets for now — revisit if calibration turns out to
        need per-bucket scales too.
        """
        logit = self.forward(planes, piece_count)
        # logit is already logit(p); centipawns = cp_scale * logit(p) is the
        # same transform as cp_scale * log(p / (1 - p)) applied to sigmoid(logit).
        return cp_scale * logit
