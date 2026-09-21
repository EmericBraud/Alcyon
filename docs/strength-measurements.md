# Strength measurements

Full record of every match played against Stockfish 8 and of the
self-play validations behind each version, with the caveats that make the
numbers readable. The README keeps only the summary table; this page keeps
the reasoning, so that a figure is never quoted without what qualifies it.


Self-play match vs. **Stockfish 8** (single-threaded, 64MB hash, no pondering, `UHO_4060_v2` opening book, [fastchess](https://github.com/Disservin/fastchess)):

| Version | Time control | Games | Chess26 score | Elo (vs SF8) |
|---|---|---|---|---|
| v5.0 | 60s+0.2s | 300 | 91.0 / 300 (30.3%) — 37W / 155L / 108D | **-169 ± 29** † |
| v5.1 | 60s+0.2s | 178 | 58.5 / 178 (32.9%) — 28W / 89L / 61D | **-124 ± 41** |
| v5.3 | 60s+0.2s | 126 | 52.0 / 126 (41.3%) — 24W / 46L / 56D | **-61 ± 34** ‡ |
| v5.4 | 60s+0.2s | 982 | 452.5 / 982 (46.1%) — 265W / 342L / 375D | **-27.3 ± 15.8** ‡§ |
| v5.5 | 60s+0.2s | 1224 | 607.5 / 1224 (49.6%) — 394W / 403L / 427D | **-2.6 ± 13.9** ‡§ |

† v5.0 was measured against a Rosetta 2 (x86_64-emulated) Stockfish 8; later
rows use a natively-compiled arm64 Stockfish 8, worth an estimated 25 Elo more.
The v5.0 figure above is its measured **-144.4 ± 28.7** shifted by that estimate
so every row in the table faces the same opponent. The shift is an estimate, not
a measurement.

‡ v5.3 was measured on a 192-core Graviton4 at concurrency 32, not on the
Mac at concurrency 3-4 like the rows above. That matters more than it sounds:
chess26 loads a 111 MB NNUE network, so running many instances in parallel
starves it of memory bandwidth in a way Stockfish 8 — whose eval is a few
kilobytes — does not. Measured at 164 concurrent games, chess26 lost 58% of
its speed against Stockfish 8's 20%, which moved the same match from -40 to
-126 Elo. Sharing the network between processes via a read-only mmap
recovered two thirds of that gap, and dropping to concurrency 32 most of the
rest, but a residual bias remains — so **-61 is a lower bound**, and the true
figure is likely better.

§ The v5.4 and v5.5 runs were stopped at 982 and 1224 games, and their
residual concurrency bias has been measured rather than estimated. Running the same
fixed-depth benchmark at 1 and at 32 simultaneous instances on the test
machine: chess26 drops from 432k to 336k nps (**-22.2%**) while Stockfish 8
goes from 2.217M to 2.342M nps (**+5.6%**, i.e. no degradation at all — its
evaluation fits in cache). The memory-mapped network removed the duplicated
memory, not the contention for bandwidth. That ~26% relative speed deficit is
worth roughly 18-25 Elo at the usual 50-70 Elo per doubling of time, so the
honest reading of this row is **around -20 Elo at an unbiased time control**.
Two caveats in the other direction: that Elo-per-doubling constant is borrowed
from the literature and has never been measured on this engine, and the
reference Stockfish 8 is built `ARCH=general-64`, without popcnt or prefetch,
which makes it weaker than a properly built one by an unmeasured amount.
OpenBench also scales the time control by machine speed, so the nominal
60s+0.2s ran as 51.6s+0.17s — identical for both engines, so no bias between
them, but not strictly the same control as the rows above.

v5.2 has no row here: it was only measured at 10s+0.1s (-52.5 ± 22.8 over
520 games), a different time control. v5.3 beat it by **+18.0 ± 7.7** over
3090 games at that control (SPRT passed, LLR 3.31).

v5.5 changes nothing in the search: it is the same engine as v5.4 with a
retrained network. The original net had only ever been trained on Stockfish's
static evaluation, never on game results, and its training had stopped at
epoch 15 with the learning rate still at 88% of its initial value — so the
weights had never been annealed either. Retraining addressed both, in two
phases from the original checkpoint: nine epochs on a blended objective
(`pt = eval_winrate * λ + game_result * (1-λ)`, λ = 0.6 constant), then four
epochs of annealing at lr 1e-4 with gamma 0.75.

Measured in self-play at 10s+0.1s, the result is **+19.0 ± 6.7** over the old
network (4386 games, SPRT passed, LLR 2.98) — and against Stockfish 8 it moves
the row from -27.3 to -2.6, essentially parity.

The decomposition is the interesting part, and it is a warning. An
intermediate network was kept after the WDL phase but before annealing, and
measured separately: annealing alone is worth **+41.0 ± 16.1**, which by
difference puts the WDL phase at **-21.9 ± 17.5**. Training on game results at
λ = 0.6 *degraded* the network; annealing recovered that and more. Three
observations pointed the same way without being connected at the time: the
validation loss oscillated instead of descending during the WDL phase, the
WDL-only network searched *more* nodes than its parent (20110 vs 18584, i.e.
less decisive evaluations), and λ = 0.6 sits at the aggressive end of the
usual 0.5-0.7 range. Annealing, by contrast, cut the epoch-to-epoch
oscillation of the validation loss by a factor of fifteen — the network
settles instead of vibrating.

The obvious follow-up, untested: annealing the *original* network with no WDL
at all should be worth around +40 rather than +19.

v5.4 has no Stockfish 8 row yet either. It adds a per-ply search stack, which
does two things: it caches the static evaluation so razoring, reverse futility
and futility share one network pass per node instead of up to three, and it
enables the `improving` heuristic by comparing a node's evaluation to its
ply-2 ancestor's. Measured against v5.3 at 10s+0.1s, the search stack and
`improving` at their default values gave **+20 Elo** over 1488 games, and a
7-parameter SPSA re-tune on top of it passed at **+18.6 Elo** over 4250 games
(LLR 2.99). The tuner's verdict on `improving` itself is worth recording: it
kept it only on late move pruning and switched it off on the other three
mechanisms, paying for it by cutting both pruning-margin slopes — the reverse
futility slope by 42%, the futility slope by 19%. Those slopes had been tuned
in a world without `improving` and were measurably off once it existed.

v5.1's run was stopped at 178 games, so its interval (±41) is wider than
v5.0's (±29) and the two overlap: the gain is the direction the numbers point,
not yet a separated result. The score comparison (32.9% vs 30.3%) is the one
figure that goes through no model at all. Most of v5.1's gain is time
management, measured in isolation at +93.9 ± 29.0 over 216 self-play games
against the commit that preceded it — a self-play gain transfers to a
different, stronger opponent at roughly half value, which is about what the
table shows.

v5.2 has no Stockfish 8 row yet: it was validated by self-play SPRT against
v5.1 instead, passing at **+103 Elo [+87, +120]** over 1140 games at a short
time control (H1 accepted, LLR 3.39 against bounds [0.00, 3.00]). Two caveats
before reading that as a table row. It is a *short* time control, and gains
that large at STC typically contract at longer ones, often by half. And it is
self-play, which transfers to a stronger opponent at roughly half value — so
the honest expectation against SF8 is a good deal less than +103, and only a
run against SF8 will say how much.

The gain comes from removing the lazy evaluation entirely. Until v5.1 the three
pruning mechanisms (razoring, reverse futility, futility) tested against the
PSQT head alone, and their margins had been SPSA-tuned against that biased
estimator. v5.2 prunes on the full network everywhere and re-tunes all nine
parameters from scratch (128,000 games). The tuner's answer contradicted the
analytical calibration that preceded it: both margin constants were driven to
or below zero and the depth slopes picked the work up instead, leaving both
margins purely proportional to depth.

Stockfish 8 is rated **~3359 Elo** on the [CCRL 40/15 list](https://ccrl.chessdom.com/ccrl/4040/rating_list_all.html). Naively offsetting that by the measured match gap gives a **very rough, unofficial estimate of ~3356 Elo** for Chess26 (v5.5) in this configuration — **this is not a CCRL rating** and shouldn't be read as one. It ignores several confounders:

- CCRL's list runs at a longer time control (40 moves/15 min) and typically multi-core, vs. our single-threaded 60+0.2 test
- The reference Stockfish 8 is built from source at tag `sf_8` with `ARCH=general-64`, so it has neither `popcnt` nor prefetch — likely *understating* its actual strength, so the true gap is probably larger
- 1224 games still leaves a wide confidence interval (±13.9 Elo from sampling alone)

An actual CCRL-comparable number would require running on CCRL's reference hardware/time control (or submitting the engine to CCRL directly, which accepts community submissions) and a fully optimized reference build for every opponent.
