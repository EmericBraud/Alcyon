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


def blended_target(score: torch.Tensor, result: torch.Tensor, lambda_: float) -> torch.Tensor:
    p_score = torch.sigmoid(score / SCORE_SCALE)
    p_result = (result + 1.0) / 2.0
    return lambda_ * p_score + (1.0 - lambda_) * p_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+", help="Path(s) to .binpack file(s)")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4, help="binpack reader concurrency")
    parser.add_argument("--lambda-start", type=float, default=1.0)
    parser.add_argument("--lambda-end", type=float, default=0.3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = ChessCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dataset = PlaneBatchDataset(
        filenames=args.binpack,
        batch_size=args.batch_size,
        cyclic=True,
        num_workers=args.num_workers,
    )
    data_iter = iter(dataset)

    import os

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
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t_start
            print(
                f"step {step + 1}/{args.steps}  loss={avg_loss:.4f}  "
                f"lambda={lambda_:.3f}  {(step + 1) * args.batch_size / elapsed:.0f} pos/s"
            )
            running_loss = 0.0

        if (step + 1) % args.checkpoint_every == 0:
            path = os.path.join(args.checkpoint_dir, f"chesscnn_step{step + 1}.pt")
            torch.save({"model": model.state_dict(), "step": step + 1}, path)
            print(f"checkpoint saved: {path}")

    final_path = os.path.join(args.checkpoint_dir, "chesscnn_final.pt")
    torch.save({"model": model.state_dict(), "step": args.steps}, final_path)
    print(f"training done, final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
