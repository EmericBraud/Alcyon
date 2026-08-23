# ♟️ Chess 26

High-Performance Chess Engine in C++

![Version](https://img.shields.io/badge/version-v5.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/language-C%2B%2B23-blue)

> Engine currently under active development.

## 💡 Project Overview

Chess 26 is a complete UCI chess engine written from scratch in modern C++, combining a **NNUE neural network evaluation** with a classical alpha-beta search. Beyond gameplay, this project is a technical showcase of low-level C++ performance work: cache-friendly data layout, SIMD, lazy incremental evaluation, and empirical (SPSA) parameter tuning via [OpenBench](https://github.com/AndyGrant/OpenBench).

It supports the UCI protocol and connects to [lichess.org](https://lichess.org) via [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) — you can play against it live at **[lichess.org/@/Chess26_BOT](https://lichess.org/@/Chess26_BOT/all)**.

## ⚙️ Technologies & Core Concepts

- **Language:** C++23
- **Board Representation:** Bitboards, with magic-bitboard/PEXT sliding-piece attack generation
- **Search:** Multithreaded alpha-beta (negamax) with iterative deepening
- **Evaluation:** NNUE (default) with an incrementally-updated accumulator, and a legacy hand-crafted evaluation (HCE) build target
- **Tuning:** SPSA parameter tuning via OpenBench; Texel tuning for the HCE evaluation
- **Endgame:** Syzygy tablebase probing via [Fathom](https://github.com/jdart1/Fathom)

## 🔬 Technical Details

### Bitboards & Move Generation

The board is represented with 64-bit bitboards, one per piece type/color. Sliding-piece attacks (bishop/rook/queen) use magic bitboards with a PEXT fast path on supporting CPUs. Move generation produces fully legal moves (castling, en passant, promotions included), filtering pseudo-legal moves by king-safety.

### Search

The search core is alpha-beta (negamax) with iterative deepening, run across multiple threads (`std::thread::hardware_concurrency()` by default). Implemented heuristics include:

- Transposition table
- Killer moves & history heuristic for move ordering
- Null-move pruning, late move reductions (LMR), aspiration windows
- Quiescence search with Static Exchange Evaluation (SEE)
- Syzygy tablebase probing in the search tree and at the root

### Evaluation: NNUE

The default evaluation is a NNUE network (Full_Threats + HalfKAv2_hm^ feature sets, L1=1024, 8 PSQT/layer-stack buckets), trained externally and quantized (int8/int16) for fast inference. On the engine side, the accumulator is:

- **Incrementally updated** per move (no full recompute), using per-piece feature toggles and a scoped threat-feature diff instead of a full-board rescan
- **Lazily materialized**: a move's accumulator update is only applied when an evaluation is actually requested, so branches that are cut off by the search (TT hits, pruning, etc.) never pay for it
- **SIMD-accelerated** (`std::experimental::simd`) with software-prefetching tuned empirically on the incremental-update hot path

A hand-crafted evaluation (material, PST, mobility, pawn structure, king safety, Texel-tuned) is available as an alternative build (`make hce`), primarily kept as a baseline/fallback and for engines/hardware where NNUE inference isn't worth its cost.

### Parameter Tuning

Search and (HCE) evaluation constants are tunable via SPSA, using [OpenBench](https://github.com/AndyGrant/OpenBench) to run distributed self-play matches and optimize parameters empirically rather than by hand — the same methodology used by Stockfish and other top engines.

## 📊 Playing Strength

Self-play match vs. **Stockfish 8** (single-threaded, 64MB hash, no pondering, `UHO_4060_v2` opening book, [fastchess](https://github.com/Disservin/fastchess)):

| Time control | Games | Chess26 score | Elo (vs SF8) |
|---|---|---|---|
| 60s+0.2s | 300 | 91.0 / 300 (30.3%) — 37W / 155L / 108D | **-144.4 ± 28.7** |

Stockfish 8 is rated **~3359 Elo** on the [CCRL 40/15 list](https://ccrl.chessdom.com/ccrl/4040/rating_list_all.html). Naively offsetting that by the measured match gap gives a **very rough, unofficial estimate of ~3215 Elo** for Chess26 in this configuration — **this is not a CCRL rating** and shouldn't be read as one. It ignores several confounders:

- CCRL's list runs at a longer time control (40 moves/15 min) and typically multi-core, vs. our single-threaded 60+0.2 test
- Stockfish 8 here ran under Rosetta 2 (x86_64 emulation on Apple Silicon), not natively — likely *understating* its actual strength, so the true gap is probably larger
- 300 games gives a fairly wide confidence interval (±28.7 Elo just from sampling)

An actual CCRL-comparable number would require running on CCRL's reference hardware/time control (or submitting the engine to CCRL directly, which accepts community submissions) and a native (non-emulated) reference build for every opponent.

## 🚀 Planned Features

- Further search enhancements (multi-cut, more selective pruning)
- Improved NNUE architecture/training pipeline
- Windows support (untested — multithreading primitives are POSIX-oriented)

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
