<p align="center">
  <img src="assets/logo.png" alt="Alcyon logo: a kingfisher diving toward the water" width="200">
</p>

# ♟️ Alcyon

High-Performance Chess Engine in C++

<sub>Formerly known as Chess26.</sub>

![Version](https://img.shields.io/badge/version-v6.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Language](https://img.shields.io/badge/language-C%2B%2B23-blue)

> Engine currently under active development.

## 💡 Project Overview

Alcyon is a complete UCI chess engine written from scratch in modern C++, combining a **NNUE neural network evaluation** with a classical alpha-beta search. Beyond gameplay, this project is a technical showcase of low-level C++ performance work: cache-friendly data layout, SIMD, lazy incremental evaluation, and empirical (SPSA) parameter tuning via [OpenBench](https://github.com/AndyGrant/OpenBench).

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

| Version | Time control | Games | LOS (vs SF8) † | Elo (vs SF8) | Estimated CCRL * |
|---|---|---|---|---|---|
| v5.0 | 60s+0.2s | 300 | <0.01% | **-169 ± 29** | ~3190 |
| v5.1 | 60s+0.2s | 178 | <0.01% | **-124 ± 41** | ~3235 |
| v5.3 | 60s+0.2s | 126 | 3.2% | **-61 ± 34** | ~3298 |
| v5.4 | 60s+0.2s | 982 | 1.4% | **-27.3 ± 15.8** | ~3332 |
| v5.5 | 60s+0.2s | 1224 | 41.1% | -2.6 ± 13.9 | ~3356 |
| v6.0 | 60s+0.2s | 820 | 97.7% | +27.2 ± 15.9 | ~3386 |
| v6.1 | 60s+0.2s | 428 | **99.1%** | **+44.1 ± 19.6** | **~3403** |
| v6.2 | 60s+0.2s | 434 (partial) | **99.8%** | **+52.4 ± 21.7** | **~3411** |
| **v6.2 + v7.nnue** ‡ | 60s+0.2s | 400 | **>99.99%** | **+134.0 ± 24.2** | **~3493** |
| **v7** § | 60s+0.2s | 240 (partial) | **>99.99%** | **+150.7 ± 31.0** | **~3510** |

v6.0 was the first version **ahead of Stockfish 8** at this time control; v6.1
adds a learned time manager; v6.2 makes the network ~44% faster with an
identical search (+46.8 ± 19.2 Elo over v6.1 in self-play, see
[docs/vitesse-nnue.md](docs/vitesse-nnue.md)).

The **v7 network** (`data/nnue/v7.nnue`) is v5 retrained for ~20 GPU hours on
filtered Leela data (test80, five months, ~16 billion positions) and annealed. With
the v6.2 search unchanged it scores **+134 against Stockfish 8** (204 wins, 57
losses, 139 draws) and +123.5 ± 17.8 against v5 in direct self-play (800 games).
See [docs/reentrainement-long.md](docs/reentrainement-long.md).

`main` also includes the low-level NNUE optimisations of
[docs/vitesse-nnue-asm.md](docs/vitesse-nnue-asm.md) (incremental threat deltas,
NEON accumulator kernels, refresh tables): **+46 % nodes/s on Apple M-series and
+54 % on x86**, with an identical search (same bench node counts). Not yet
measured in games.

‡ Measured on a rented 64-core server (concurrency 30) rather than on the Mac.
v6.2 replayed there as a reference scored +25.2 ± 25.1, so the network alone is
worth about +109 Elo against Stockfish 8.

§ v7 = the v7 network plus the NNUE speed work below (`main` at fe9afc6). Same
server, but at concurrency 16 while a network training ran on the same machine,
stopped after 240 games (132 wins, 34 losses, 74 draws). All 7 time forfeits were
Stockfish 8's, most likely a side effect of the load: counting them as draws gives
about +139. Read it as "at least as strong as v6.2 + v7.nnue", not as a precise
measure of the speed gain.

† LOS — likelihood of superiority — is the probability that Alcyon is the
stronger engine, from the decisive games only: `Φ((W−L)/√(2(W+L)))`. It answers
"could this gap be luck?" where the Elo column answers "how big is it?". At 41%
v5.5 is statistically indistinguishable from Stockfish 8; below 5% the gap is
real.

\* The CCRL column offsets Stockfish 8's
[CCRL 40/15](https://ccrl.chessdom.com/ccrl/4040/rating_list_all.html) rating of
~3359 by the measured gap. It is **not a CCRL rating**: CCRL runs a longer time
control on different hardware, and the number inherits every caveat of the match
it comes from. Two of those caveats both point the same way — the reference
Stockfish 8 is built `ARCH=general-64`, without popcnt or prefetch, and the
matches run at concurrency 32, which costs Alcyon more speed than Stockfish (a
111 MB network against a few-kilobyte evaluation). So if anything these figures
understate the engine.

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
git clone https://github.com/EmericBraud/alcyon.git
cd alcyon
make        # builds ./alcyon (NNUE + SPSA tuning enabled by default)
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
./alcyon
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
Alcyon's networks are trained on positions from [Leela Chess Zero](https://lczero.org/)'s self-play games: many thanks to the Lc0 project and its contributors for releasing their [training data](https://storage.lczero.org/files/training_data/) openly, and to [linrock](https://huggingface.co/linrock) for filtering and converting it into the test80 binpacks used to train the v7 network. The networks are trained with [nnue-pytorch](https://github.com/official-stockfish/nnue-pytorch).
Finally, thanks to the [Stockfish](https://github.com/official-stockfish/stockfish) and [Ethereal](https://github.com/AndyGrant/Ethereal) communities for developing and sharing such performant chess engines.

## 👨‍💻 Author

Emeric Braud
https://www.linkedin.com/in/emeric-braud-101239151/
