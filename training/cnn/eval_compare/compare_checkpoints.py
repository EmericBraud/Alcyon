"""Compare CNN checkpoints against NNUE and Stockfish, on an independent
test set (wac.epd — "Win At Chess", 299 well-known tactical positions
from albertoruibal/carballo, not part of our training data).

Methodology (see docs/gpu-async-eval/ for the full discussion):
  - Ground truth is Stockfish's own STATIC eval ("eval" UCI debug
    command, no search) — NOT its searched score. WAC positions are
    tactical by construction (a quiet-looking surface hiding a forced
    combination a deep search resolves), so comparing a static
    evaluator against a *searched* score measures search depth, not
    eval quality. Both NNUE and CNN are static evaluators here, so the
    fair reference is Stockfish's static eval.
  - Correlation is the primary metric — scale-invariant, meaningful
    even though CNN's cp_scale (SCORE_SCALE in train.py) has never
    been calibrated. MAE is kept for reference but is noisier and
    partly reflects that uncalibrated scale, not model quality — don't
    over-read a MAE change without also checking correlation.
  - Forced-mate positions and in-check positions (where Stockfish's
    static eval is undefined, "none (in check)") are excluded from
    both metrics.

Stockfish and NNUE scores are cached (they never change — only the
CNN checkpoints do) so re-running after a new checkpoint doesn't
recompute the slow reference lasts several minutes at depth 15.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

import chess
import chess.engine
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import ChessCNN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EPD = os.path.join(HERE, "wac.epd")
DEFAULT_CACHE = os.path.join(HERE, "ground_truth_cache.json")
MATE_SENTINEL_CP = 10000

NUM_PLANES, BOARD_SIZE = 19, 8
PIECE_TYPE_INDEX = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
                     chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


def flip_rank(sq):
    return sq ^ 56


def fen_to_planes_and_piece_count(fen):
    """Mirrors plane_batch.cpp's layout exactly — keep the two in sync."""
    board = chess.Board(fen)
    us = board.turn
    them = not us
    orient_flip = not us

    planes = torch.zeros(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
    flat = planes.view(NUM_PLANES, 64)

    non_king_pieces = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        is_us = piece.color == us
        plane = PIECE_TYPE_INDEX[piece.piece_type] + (0 if is_us else 6)
        oriented_sq = flip_rank(sq) if orient_flip else sq
        flat[plane, oriented_sq] = 1.0
        if piece.piece_type != chess.KING:
            non_king_pieces += 1

    if us == chess.WHITE:
        flat[12, :] = 1.0
    if board.has_kingside_castling_rights(us):
        flat[13, :] = 1.0
    if board.has_queenside_castling_rights(us):
        flat[14, :] = 1.0
    if board.has_kingside_castling_rights(them):
        flat[15, :] = 1.0
    if board.has_queenside_castling_rights(them):
        flat[16, :] = 1.0
    if board.ep_square is not None:
        oriented_ep = flip_rank(board.ep_square) if orient_flip else board.ep_square
        flat[17, oriented_ep] = 1.0
    flat[18, :] = board.halfmove_clock / 100.0

    return planes, non_king_pieces


def parse_epd_fens(path):
    fens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fields = line.split()
            # EPD gives 4 FEN fields (board, side, castling, ep) plus
            # "bm ...; id ...;" — append default halfmove/fullmove.
            fens.append(" ".join(fields[:4]) + " 0 1")
    return fens


def stockfish_searched_scores(fens, stockfish_bin, depth):
    scores = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_bin) as engine:
        for fen in fens:
            board = chess.Board(fen)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].pov(board.turn)
            if score.is_mate():
                sign = 1 if score.mate() > 0 else -1
                scores.append(sign * MATE_SENTINEL_CP)
            else:
                scores.append(score.score())
    return scores


def stockfish_static_scores(fens, stockfish_bin):
    """Stockfish's own static eval (no search), via its "eval" debug
    command — always printed from White's perspective, converted here
    to side-to-move perspective to match NNUE/CNN's convention. None
    for in-check positions (static eval undefined there)."""
    proc = subprocess.Popen(
        [stockfish_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    scores = []
    for fen in fens:
        board = chess.Board(fen)
        proc.stdin.write(f"position fen {fen}\n")
        proc.stdin.write("eval\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        while not line.startswith("Final evaluation"):
            line = proc.stdout.readline()
        token = line.split()[2]
        if token == "none":
            scores.append(None)
            continue
        pawns = float(token)
        cp_white = round(pawns * 100)
        scores.append(cp_white if board.turn == chess.WHITE else -cp_white)
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)
    return scores


def nnue_scores(fens, nnue_bin):
    """chess26's own NNUE build, "eval" UCI debug command — static
    eval, no search, from the side-to-move's perspective."""
    proc = subprocess.Popen(
        [nnue_bin], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    scores = []
    for fen in fens:
        proc.stdin.write(f"position fen {fen}\n")
        proc.stdin.write("eval\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        while not line.startswith("score:"):
            line = proc.stdout.readline()
        scores.append(int(line.split(":")[1].strip()))
    proc.stdin.write("quit\n")
    proc.stdin.flush()
    proc.wait(timeout=5)
    return scores


def load_or_compute_ground_truth(epd_path, cache_path, stockfish_bin, nnue_bin, depth, force):
    if not force and os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("epd_path") == os.path.abspath(epd_path) and cache.get("depth") == depth:
            print(f"using cached ground truth: {cache_path}", flush=True)
            return cache

    fens = parse_epd_fens(epd_path)
    print(f"loaded {len(fens)} positions from {epd_path}", flush=True)

    print(f"running Stockfish searched score (depth={depth})...", flush=True)
    sf = stockfish_searched_scores(fens, stockfish_bin, depth)
    print("running Stockfish static eval...", flush=True)
    sf_static = stockfish_static_scores(fens, stockfish_bin)
    print("running NNUE eval...", flush=True)
    nnue = nnue_scores(fens, nnue_bin)

    cache = {
        "epd_path": os.path.abspath(epd_path),
        "depth": depth,
        "fens": fens,
        "stockfish_searched": sf,
        "stockfish_static": sf_static,
        "nnue": nnue,
    }
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"cached ground truth: {cache_path}", flush=True)
    return cache


def cnn_scores(fens_planes, piece_counts, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = ChessCNN().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        pred_cp = model.eval_centipawns(fens_planes, piece_counts, cp_scale=410.0)
    return pred_cp.tolist(), ckpt["step"]


def correlation(a, b):
    a_t, b_t = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    a_c, b_c = a_t - a_t.mean(), b_t - b_t.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-9)).item()


def mae(a, b):
    a_t, b_t = torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32)
    return (a_t - b_t).abs().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epd", default=DEFAULT_EPD)
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="ground-truth cache JSON path")
    parser.add_argument("--force-recompute", action="store_true",
                         help="ignore the cache and recompute Stockfish/NNUE scores")
    parser.add_argument("--stockfish-bin", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--nnue-bin", required=True,
                         help="path to chess26's NNUE-enabled UCI binary (build-nnue/chess26)")
    parser.add_argument("--checkpoints", required=True, help="glob pattern for CNN checkpoints, e.g. 'checkpoints/chesscnn_step*.pt'")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish search depth for the (reference-only) searched score")
    parser.add_argument("--out", default=None, help="write the CNN learning-curve JSON here")
    args = parser.parse_args()

    device = torch.device("cpu")

    gt = load_or_compute_ground_truth(
        args.epd, args.cache, args.stockfish_bin, args.nnue_bin, args.depth, args.force_recompute
    )
    fens = gt["fens"]
    sf, sf_static, nnue = gt["stockfish_searched"], gt["stockfish_static"], gt["nnue"]

    keep = [abs(s) < MATE_SENTINEL_CP and static is not None
            for s, static in zip(sf, sf_static)]
    sf_static_f = [s for s, k in zip(sf_static, keep) if k]
    nnue_f = [s for s, k in zip(nnue, keep) if k]
    print(f"{sum(keep)}/{len(fens)} positions kept for comparison "
          f"(mate/in-check positions excluded)", flush=True)

    nnue_mae, nnue_corr = mae(nnue_f, sf_static_f), correlation(nnue_f, sf_static_f)
    print(f"NNUE reference: MAE={nnue_mae:.1f}  corr={nnue_corr:.4f}\n", flush=True)

    print("encoding planes once for all positions...", flush=True)
    all_planes, all_pc = [], []
    for fen in fens:
        planes, pc = fen_to_planes_and_piece_count(fen)
        all_planes.append(planes)
        all_pc.append(pc)
    planes_batch = torch.stack(all_planes).to(device)
    pc_batch = torch.tensor(all_pc, dtype=torch.int64, device=device)

    ckpt_paths = sorted(
        glob.glob(args.checkpoints),
        key=lambda p: int(re.search(r"step(\d+)", p).group(1)),
    )
    if not ckpt_paths:
        print(f"no checkpoints matched: {args.checkpoints}")
        return

    curve = []
    for ckpt_path in ckpt_paths:
        cnn, step = cnn_scores(planes_batch, pc_batch, ckpt_path, device)
        cnn_f = [s for s, k in zip(cnn, keep) if k]
        m, c = mae(cnn_f, sf_static_f), correlation(cnn_f, sf_static_f)
        curve.append({"step": step, "mae": m, "corr": c})
        print(f"step {step:6d}  MAE={m:8.1f}  corr={c:.4f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cnn": curve, "nnue": {"mae": nnue_mae, "corr": nnue_corr}}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
