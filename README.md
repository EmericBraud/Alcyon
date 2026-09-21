# ♟️ Chess 26

High-Performance Chess Engine in C++

![Version](https://img.shields.io/badge/version-v5.5-blue)
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

Self-play match vs. **Stockfish 8** (single-threaded, 64MB hash, no pondering,
`UHO_4060_v2` opening book, [fastchess](https://github.com/Disservin/fastchess)):

| Version | Time control | Games | Chess26 score | Elo (vs SF8) |
|---|---|---|---|---|
| v5.0 | 60s+0.2s | 300 | 30.3% | **-169 ± 29** |
| v5.1 | 60s+0.2s | 178 | 32.9% | **-124 ± 41** |
| v5.3 | 60s+0.2s | 126 | 41.3% | **-61 ± 34** |
| v5.4 | 60s+0.2s | 982 | 46.1% | **-27.3 ± 15.8** |
| v5.5 | 60s+0.2s | 1224 | 49.6% | **-2.6 ± 13.9** |

v5.5 plays **on par with Stockfish 8** at this time control. Two caveats worth
stating up front: the reference Stockfish 8 is built `ARCH=general-64`, without
popcnt or prefetch, which understates it; and the matches run at concurrency 32,
which costs chess26 more speed than Stockfish (a 111 MB network against a
few-kilobyte evaluation), so the figure is if anything pessimistic.

**[docs/strength-measurements.md](docs/strength-measurements.md)** records every
match in full — what each version changed, the self-play SPRTs behind it, and the
measured biases. Including the negative results: a network trained on game
results lost 22 Elo, and annealing its weights won back 41.

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
