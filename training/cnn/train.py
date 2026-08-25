"""Training loop for the CNN position evaluator.

Target construction follows the same lambda-annealed blend used for
Stockfish NNUE training: early steps weight the (locally precise, but
Stockfish-biased) search score heavily; later steps shift weight
toward the (noisier, but bias-correcting) actual game result. See
docs/gpu-async-eval/ for the design discussion.

No special-casing is needed for mate-score sentinels (e.g. score ==
32002 in the binpack): sigmoid(score / SCORE_SCALE) saturates toward
0/1 for those magnitudes on its own, as long as the loss is computed
in a numerically stable way (binary_cross_entropy_with_logits, not a
manual sigmoid followed by log).
"""

import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from data_loader import PlaneBatchDataset
from model import ChessCNN

# Scale used to convert a raw Stockfish score into a win probability
# via sigmoid(score / SCORE_SCALE). Same family/order of magnitude as
# ChessCNN.eval_centipawns' cp_scale, but this one is a training
# hyperparameter for target construction — recalibrate independently
# once real training runs show whether it matches this dataset's
# score distribution.
SCORE_SCALE = 410.0


def lambda_schedule(step: int, total_steps: int, lambda_start: float, lambda_end: float) -> float:
    progress = min(step / max(total_steps, 1), 1.0)
    return lambda_start + (lambda_end - lambda_start) * progress


def lr_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup, then cosine decay to ~0. Returns a multiplier in [0, 1]
    for torch.optim.lr_scheduler.LambdaLR (applied on top of --lr)."""
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def blended_target(score: torch.Tensor, result: torch.Tensor, lambda_: float) -> torch.Tensor:
    p_score = torch.sigmoid(score / SCORE_SCALE)
    p_result = (result + 1.0) / 2.0
    return lambda_ * p_score + (1.0 - lambda_) * p_result


@torch.no_grad()
def evaluate(model, val_iter, device, num_batches, lambda_):
    """Runs num_batches from val_iter and returns the mean blended-target loss.

    lambda_ is fixed at the current training step's value so the
    validation loss is comparable to the training loss logged at the
    same point, not distorted by an annealing mismatch.
    """
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        planes, score, result, piece_count = next(val_iter)
        planes = planes.to(device)
        score = score.to(device).squeeze(-1)
        result = result.to(device).squeeze(-1)
        piece_count = piece_count.to(device)

        target = blended_target(score, result, lambda_)
        logits = model(planes, piece_count)
        total_loss += F.binary_cross_entropy_with_logits(logits, target).item()
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+", help="Path(s) to .binpack file(s)")
    parser.add_argument(
        "--val-binpack",
        nargs="+",
        default=None,
        help="Path(s) to a held-out .binpack for validation. If omitted, "
        "no validation loss is reported — only the training loss, which "
        "cannot tell you whether the model generalizes or is overfitting.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--val-batch-size", type=int, default=4096)
    parser.add_argument("--val-batches", type=int, default=20, help="batches per validation pass")
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm, 0 to disable")
    parser.add_argument("--num-workers", type=int, default=4, help="binpack reader concurrency")
    parser.add_argument("--lambda-start", type=float, default=1.0)
    parser.add_argument("--lambda-end", type=float, default=0.3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.val_binpack is None:
        print(
            "WARNING: no --val-binpack given. Only training loss will be "
            "logged — you won't be able to tell overfitting from real "
            "progress. Pass a held-out .binpack once you have one.",
            flush=True,
        )

    device = torch.device(args.device)
    model = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_schedule(step, args.steps, args.warmup_steps)
    )

    dataset = PlaneBatchDataset(
        filenames=args.binpack,
        batch_size=args.batch_size,
        cyclic=True,
        num_workers=args.num_workers,
    )
    data_iter = iter(dataset)

    val_iter = None
    if args.val_binpack is not None:
        val_dataset = PlaneBatchDataset(
            filenames=args.val_binpack,
            batch_size=args.val_batch_size,
            cyclic=True,  # cyclic so a short/small val set never raises StopIteration mid-run
            num_workers=1,
        )
        val_iter = iter(val_dataset)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    running_loss = 0.0
    t_start = time.time()

    for step in range(args.steps):
        planes, score, result, piece_count = next(data_iter)
        planes = planes.to(device)
        score = score.to(device).squeeze(-1)
        result = result.to(device).squeeze(-1)
        piece_count = piece_count.to(device)

        lambda_ = lambda_schedule(step, args.steps, args.lambda_start, args.lambda_end)
        target = blended_target(score, result, lambda_)

        logits = model(planes, piece_count)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t_start
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step {step + 1}/{args.steps}  loss={avg_loss:.4f}  "
                f"lambda={lambda_:.3f}  lr={current_lr:.2e}  "
                f"{(step + 1) * args.batch_size / elapsed:.0f} pos/s"
            )
            running_loss = 0.0

        if val_iter is not None and (step + 1) % args.val_every == 0:
            val_loss = evaluate(model, val_iter, device, args.val_batches, lambda_)
            print(f"step {step + 1}/{args.steps}  val_loss={val_loss:.4f}")

        if (step + 1) % args.checkpoint_every == 0:
            path = os.path.join(args.checkpoint_dir, f"chesscnn_step{step + 1}.pt")
            torch.save({"model": model.state_dict(), "step": step + 1}, path)
            print(f"checkpoint saved: {path}")

    final_path = os.path.join(args.checkpoint_dir, "chesscnn_final.pt")
    torch.save({"model": model.state_dict(), "step": args.steps}, final_path)
    print(f"training done, final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
