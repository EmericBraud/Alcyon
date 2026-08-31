# Eval comparison: CNN vs NNUE vs Stockfish

## results_log.csv — the source of truth for every correlation number

**Convention: every WDL/score-recherché correlation measurement quoted
anywhere (chat, docs, commit messages) MUST be appended as a row to
`results_log.csv` in this directory, and read from there when citing a
past number — never from memory.** This exists because re-typing
numbers from memory across a long session is exactly how silent
transcription errors creep in.

Row fields: `date, model, checkpoint_step, architecture, params,
num_positions, device, num_workers, nnue_score_corr, nnue_wdl_corr,
model_score_corr, model_wdl_corr, notes`. `num_workers` matters —
see `eval_compare_sampling_noise` in project memory: with
`num_workers > 1` the sample is non-deterministic across runs, so two
measurements of the "same" checkpoint can legitimately differ by
+/-0.005-0.01 purely from sampling. Always prefer `num_workers=1`
(the default) for numbers meant to be compared across time; note the
sample size too, since a smaller `--num-positions` widens that same
noise band.

`compare_checkpoints.py` measures how well the CNN checkpoints predict
Stockfish's own **static** eval (no search), on `wac.epd` — 299
positions from the classic "Win At Chess" test suite (source:
[albertoruibal/carballo](https://github.com/albertoruibal/carballo)),
independent of the training binpack.

Correlation is the metric to trust — it's scale-invariant, so it
stays meaningful even though the CNN's `cp_scale` (`SCORE_SCALE` in
`train.py`) has never been calibrated. MAE is kept for reference but
is noisier and partly reflects that uncalibrated scale rather than
model quality.

See `compare_checkpoints.py`'s module docstring for the full
methodology, in particular **why the reference is Stockfish's static
eval, not its searched score** (WAC positions are tactical by
construction — a searched score already resolves combinations no
static evaluator, NNUE or CNN, can see).

## Usage

```bash
python3 compare_checkpoints.py \
  --nnue-bin /path/to/chess26/build-nnue/chess26 \
  --checkpoints "checkpoints/chesscnn_step*.pt" \
  --out learning_curve.json
```

First run computes and caches Stockfish/NNUE scores in
`ground_truth_cache.json` (checked in — small, deterministic, and
expensive to recompute at depth 15). Later runs with new checkpoints
reuse the cache automatically; pass `--force-recompute` to redo it
(e.g. after changing `--depth` or the epd file).

Requires a local `stockfish` binary (`--stockfish-bin`, defaults to
the Homebrew path) and `python-chess`.
