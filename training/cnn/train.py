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
import warnings

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
def evaluate(model, val_iter, device, num_batches, lambda_, autocast_dtype):
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
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = model(planes, piece_count)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        total_loss += loss.item()
    model.train()
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binpack", required=True, nargs="+", help="Path(s) to .binpack file(s)")
    parser.add_argument(
        "--val-binpack",
        nargs="+",
        default=None,
        help="Path(s) to a SEPARATE held-out .binpack for validation, if "
        "you have one. If omitted, validation positions are instead "
        "carved out of --binpack itself via a deterministic per-position "
        "hash split (--val-percent) — see plane_batch_stream.h. No file "
        "is physically cut; the split is applied while streaming.",
    )
    parser.add_argument(
        "--val-percent",
        type=int,
        default=5,
        help="Percent of --binpack positions held out for validation via "
        "the hash split, used only when --val-binpack is not given. 0 "
        "disables validation entirely.",
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
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint .pt to resume from — restores model "
        "weights, the training step counter, and the LR schedule "
        "position. Optimizer momentum (Adam's per-parameter moment "
        "estimates) is also restored if the checkpoint has it; older "
        "checkpoints saved before this feature only have model+step, "
        "in which case the optimizer restarts cold (a brief LR-schedule "
        "hiccup, not a correctness issue).",
    )
    parser.add_argument(
        "--extend-steps",
        type=int,
        default=None,
        help="Continue training past a FINISHED run's horizon (its cosine "
        "decay already hit lr=0 and lambda already hit --lambda-end, so a "
        "plain --resume-from with a bigger --steps would either train at "
        "lr=0 or corrupt the lambda/LR curves). Requires --resume-from. "
        "Runs a fresh mini warmup+cosine-decay over this many additional "
        "steps, with lambda held fixed at --lambda-end throughout (no "
        "further annealing beyond the floor already reached).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help="torch.set_num_threads() cap. PyTorch defaults this to the "
        "visible host core count, which on a cgroup-limited container "
        "(shared GPU rental, Docker, etc.) can be wildly higher than the "
        "actual CPU quota — causing massive thread oversubscription and "
        "contention rather than useful parallelism. Set to comfortably "
        "under the container's real CPU allocation.",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(max(2, args.cpu_threads // 4))

    # Input shape (planes, batch_size) never changes across steps, so
    # cuDNN's autotuner can safely pick the fastest conv algorithm once
    # and reuse it — free speedup, no correctness tradeoff for a
    # fixed-shape workload like this one.
    torch.backends.cudnn.benchmark = True

    use_hash_split = args.val_binpack is None and args.val_percent > 0
    if args.val_binpack is None and args.val_percent <= 0:
        print(
            "WARNING: no --val-binpack and --val-percent <= 0. Only "
            "training loss will be logged — you won't be able to tell "
            "overfitting from real progress.",
            flush=True,
        )

    print(f"torch {torch.__version__}, device={args.device}", flush=True)

    if args.extend_steps is not None and args.resume_from is None:
        raise SystemExit("--extend-steps requires --resume-from")

    device = torch.device(args.device)
    print("building model...", flush=True)
    model = ChessCNN().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"model ready: {num_params:,} params on {device}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_step = 0
    if args.resume_from is not None:
        print(f"resuming from checkpoint: {args.resume_from}", flush=True)
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        else:
            print(
                "WARNING: checkpoint has no optimizer state (saved before "
                "--resume-from existed) — Adam restarts cold, expect a "
                "brief hiccup in the LR schedule's effective step size.",
                flush=True,
            )
        print(f"resumed at step {start_step}", flush=True)

    # bf16 autocast: same dynamic range as fp32 (just less mantissa),
    # so it's safe without a GradScaler — free throughput on GPUs with
    # bf16 tensor cores. Only enabled on CUDA; MPS/CPU autocast support
    # is inconsistent, not worth the risk for a "free" optimization.
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    # torch.compile fuses the trunk's conv/BN/ReLU chain into fewer
    # kernel launches. Compiling the raw (uncompiled) `model` object
    # for training — model.state_dict()/checkpointing below stays
    # untouched by this, since torch.compile wraps a module without
    # replacing it; `train_model` is only used for the forward call.
    train_model = torch.compile(model) if device.type == "cuda" else model

    extending = args.extend_steps is not None
    if extending:
        args.steps = start_step + args.extend_steps
        print(
            f"extending past the original horizon: {args.extend_steps} new steps "
            f"(fresh warmup+decay), lambda held fixed at {args.lambda_end}",
            flush=True,
        )
        # Fresh, independent mini-schedule for just the extension —
        # deliberately NOT a continuation of the original cosine curve
        # (that one already finished, decayed to lr=0). A fresh scheduler
        # (last_epoch=-1) naturally counts its OWN .step() calls from 0,
        # which lines up with lr_schedule(local_step, extend_steps, ...)
        # without any fast-forward replay needed.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: lr_schedule(step, args.extend_steps, args.warmup_steps)
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: lr_schedule(step, args.steps, args.warmup_steps)
        )
        if start_step > 0:
            # Fast-forward the scheduler's internal counter to start_step
            # so the next .step() call resumes the cosine decay at the
            # right point, instead of restarting warmup from step 0. This
            # deliberately calls scheduler.step() without a matching
            # optimizer.step() in between (we're only replaying the
            # counter, not re-doing the optimization), which triggers a
            # harmless PyTorch ordering warning — silenced here since it
            # doesn't apply to this use.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(start_step):
                    scheduler.step()

    print(f"opening binpack stream: {args.binpack}", flush=True)
    # When splitting a single file by hash, the training stream must
    # skip the validation share (val_percent, is_validation=False) so
    # the two streams never see the same position.
    dataset = PlaneBatchDataset(
        filenames=args.binpack,
        batch_size=args.batch_size,
        cyclic=True,
        num_workers=args.num_workers,
        val_percent=args.val_percent if use_hash_split else 0,
        is_validation=False,
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
    elif use_hash_split:
        val_dataset = PlaneBatchDataset(
            filenames=args.binpack,
            batch_size=args.val_batch_size,
            cyclic=True,
            num_workers=1,
            val_percent=args.val_percent,
            is_validation=True,
        )
        val_iter = iter(val_dataset)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"checkpoints will be written to: {os.path.abspath(args.checkpoint_dir)}", flush=True)
    print(f"starting training: {args.steps} steps, batch_size={args.batch_size}", flush=True)

    running_loss = 0.0
    t_start = time.time()
    t_last_log = t_start

    for step in range(start_step, args.steps):
        planes, score, result, piece_count = next(data_iter)
        planes = planes.to(device)
        score = score.to(device).squeeze(-1)
        result = result.to(device).squeeze(-1)
        piece_count = piece_count.to(device)

        lambda_ = args.lambda_end if extending else lambda_schedule(
            step, args.steps, args.lambda_start, args.lambda_end
        )
        target = blended_target(score, result, lambda_)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            logits = train_model(planes, piece_count)
            loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        # Print the very first step unconditionally — with a large
        # batch size, waiting for the first --log-every window can
        # look like the process has hung when it's actually just
        # compiling MPS/CUDA kernels or loading the first batches.
        if step == start_step or (step + 1) % args.log_every == 0:
            steps_since_log = 1 if step == start_step else args.log_every
            avg_loss = running_loss / steps_since_log
            now = time.time()
            window_pos_per_s = (steps_since_log * args.batch_size) / max(now - t_last_log, 1e-9)
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"step {step + 1}/{args.steps}  loss={avg_loss:.4f}  "
                f"lambda={lambda_:.3f}  lr={current_lr:.2e}  "
                f"{window_pos_per_s:.0f} pos/s  ({now - t_start:.0f}s elapsed)",
                flush=True,
            )
            running_loss = 0.0
            t_last_log = now

        if val_iter is not None and (step + 1) % args.val_every == 0:
            val_loss = evaluate(train_model, val_iter, device, args.val_batches, lambda_, autocast_dtype)
            print(f"step {step + 1}/{args.steps}  val_loss={val_loss:.4f}", flush=True)

        if (step + 1) % args.checkpoint_every == 0:
            path = os.path.join(args.checkpoint_dir, f"chesscnn_step{step + 1}.pt")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step + 1},
                path,
            )
            print(f"checkpoint saved: {path}", flush=True)

    final_path = os.path.join(args.checkpoint_dir, "chesscnn_final.pt")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": args.steps},
        final_path,
    )
    print(f"training done, final checkpoint: {final_path}", flush=True)


if __name__ == "__main__":
    main()
