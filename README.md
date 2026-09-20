# ♟️ Chess 26

High-Performance Chess Engine in C++

![Version](https://img.shields.io/badge/version-v5.4-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/language-C%2B%2B23-blue)

> Engine currently under active development.

## 💡 Project Overview

Chess 26 is a complete UCI chess engine written from scratch in modern C++, combining a **NNUE neural network evaluation** with a classical alpha-beta search. Beyond gameplay, this project is a technical showcase of low-level C++ performance work: cache-friendly data layout, SIMD, lazy incremental evaluation, and empirical (SPSA) parameter tuning via [OpenBench](https://github.com/AndyGrant/OpenBench).

It supports the UCI protocol and connects to [lichess.org](https://lichess.org) via [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) — you can play against it live at **[lichess.org/@/Chess26_BOT](https://lichess.org/@/Chess26_BOT/all)**.

📖 **[Project wiki — emericbraud.github.io](https://emericbraud.github.io)** — write-ups of what
I learned building this engine: the techniques, the measurements behind them, and what did and
didn't work.

## ⚙️ Technologies & Core Concepts

- **Language:** C++23
- **Board Representation:** Bitboards, with magic-bitboard/PEXT sliding-piece attack generation
- **Search:** Multithreaded alpha-beta (negamax) with iterative deepening
- **Evaluation:** NNUE (default) with an incrementally-updated accumulator, and a legacy hand-crafted evaluation (HCE) build target
- **Tuning:** SPSA parameter tuning via OpenBench; Texel tuning for the HCE evaluation
- **Endgame:** Syzygy tablebase probing via [Fathom](https://github.com/jdart1/Fathom)

## 📊 Playing Strength

Self-play match vs. **Stockfish 8** (single-threaded, 64MB hash, no pondering, `UHO_4060_v2` opening book, [fastchess](https://github.com/Disservin/fastchess)):

| Version | Time control | Games | Chess26 score | Elo (vs SF8) |
|---|---|---|---|---|
| v5.0 | 60s+0.2s | 300 | 91.0 / 300 (30.3%) — 37W / 155L / 108D | **-169 ± 29** † |
| v5.1 | 60s+0.2s | 178 | 58.5 / 178 (32.9%) — 28W / 89L / 61D | **-124 ± 41** |
| v5.3 | 60s+0.2s | 126 | 52.0 / 126 (41.3%) — 24W / 46L / 56D | **-61 ± 34** ‡ |
| v5.4 | 60s+0.2s | 644 | 282.5 / 644 (43.9%) — 161W / 240L / 243D | **-42.8 ± 20.0** ‡§ |

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

§ The v5.4 run is still in progress (644 of 2000 games) and its residual
concurrency bias has been measured rather than estimated. Running the same
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

Stockfish 8 is rated **~3359 Elo** on the [CCRL 40/15 list](https://ccrl.chessdom.com/ccrl/4040/rating_list_all.html). Naively offsetting that by the measured match gap gives a **very rough, unofficial estimate of ~3235 Elo** for Chess26 (v5.1) in this configuration — **this is not a CCRL rating** and shouldn't be read as one. It ignores several confounders:

- CCRL's list runs at a longer time control (40 moves/15 min) and typically multi-core, vs. our single-threaded 60+0.2 test
- The reference Stockfish 8 is built from source at tag `sf_8` with `ARCH=general-64`, so it has neither `popcnt` nor prefetch — likely *understating* its actual strength, so the true gap is probably larger
- 178 games gives a wide confidence interval (±41 Elo just from sampling)

An actual CCRL-comparable number would require running on CCRL's reference hardware/time control (or submitting the engine to CCRL directly, which accepts community submissions) and a fully optimized reference build for every opponent.

## 🛠️ Build & Run

### Requirements

- A C++23-compatible compiler (tested with **g++** on Linux x86 and macOS ARM; not yet tested on Windows)
- CMake

### Build Instructions

```bash
git clone https://github.com/EmericBraud/chess26.git
cd chess26
make        # builds ./chess26 (NNUE + SPSA tuning enabled by default)
```

Other Makefile targets:

```bash
make nnue       # NNUE build only
make hce        # HCE-only build (NNUE disabled)
make test-nnue  # build NNUE + run the unit test suite
make test-hce   # build HCE + run the unit test suite
```

Run the engine (UCI protocol):

```bash
./chess26
```

### Running Tests

```bash
ctest --test-dir build --output-on-failure
```

## 📝 License

This project is licensed under the MIT License.

### Dependencies

This project depends on:

- Fathom: a C project developed by jdart1 that helps probing Syzygy tablebases — https://github.com/jdart1/Fathom (MIT license)
- SFML (if `ENABLE_GUI` is set): a C++ GUI library developed by Laurent Gomila — https://github.com/SFML/SFML (Zlib license)
- Google Test — https://github.com/google/googletest (BSD-3-Clause license)

### Credits

I want to thank the [chessprogramming.org](https://www.chessprogramming.org/) community for sharing such precious information.
Thanks to the [OpenBench](https://github.com/andygrant/openbench) creators for sharing such an amazing SPSA tuning tool.
I also greatly used [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) to connect my UCI engine to lichess.org.
The [Texel](https://github.com/peterosterlund2/texel) tuning technique also greatly helped me improve the HCE function.
Finally, thanks to the [Stockfish](https://github.com/official-stockfish/stockfish) and [Ethereal](https://github.com/AndyGrant/Ethereal) communities for developing and sharing such performant chess engines.

## 👨‍💻 Author

Emeric Braud
https://www.linkedin.com/in/emeric-braud-101239151/
