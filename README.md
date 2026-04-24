# ♟️ Chess 26

High-Performance Chess Engine in C++

> Engine currently under active development.

## 💡 Project Overview

Chess 26 is an ambitious project aimed at building a complete and efficient chess engine from scratch. The primary goal is to develop a program capable of playing chess at a strong level while maintaining high execution speed and computational efficiency.

Beyond gameplay, this project serves as a technical showcase of modern C++ expertise, low-level optimization techniques, and algorithmic design required for performance-critical systems.

## ⚙️ Technologies & Core Concepts

This engine is written entirely in modern C++ (C++23) and focuses heavily on performance-oriented design.

- **Language:** C++23
- **Architecture:** Performance-driven, cache-friendly design
- **Board Representation:** Bitboards for compact and fast state manipulation
- **Search Algorithm:** Minimax with Alpha-Beta pruning

## 🔬 Technical Details

### Bitboards

The engine uses 64-bit bitboards to represent the chessboard. Each piece type and color is encoded as a separate bitboard, allowing extremely fast operations using bitwise instructions.

This enables:

- Efficient move generation
- Fast attack detection
- Minimal memory footprint
- High cache locality

### Move Generation

Legal move generation is fully implemented, including:

- Castling
- En passant
- Promotions

The engine ensures legality by filtering pseudo-legal moves and verifying king safety.

### Search

The core search algorithm is based on:

- **Minimax**
- **Alpha-Beta pruning**

Key optimizations include:

- Move ordering to improve pruning efficiency
- Iterative deepening (if implemented/added later)
- Efficient evaluation accumulation

### Heuristics & Optimizations

To reduce the search space and improve performance, several classic heuristics are implemented:

- **Transposition Table**  
  Caches previously evaluated positions using hashing to avoid redundant computation.

- **Killer Moves Heuristic**  
  Prioritizes moves that previously caused beta cutoffs.

- **History Heuristic**  
  Tracks move effectiveness over time to improve ordering.

- **Efficient Memory Usage**  
  Careful data layout and use of low-level constructs to maximize CPU efficiency.

### Evaluation Function

The evaluation function is based on:

- Material balance
- Piece-square tables (PST)
- Mobility
- Pawn structure
- King safety

It is designed to be:

- Fast (called millions of times per search)
- Incrementally computable where possible

## 🚀 Planned Features

- Play full games against the engine
- FEN (Forsyth-Edwards Notation) support
- Command-line interface for interaction
- Improved evaluation and search enhancements
- Potential UCI protocol support

## 🛠️ Build & Run

### Requirements

- A C++23-compatible compiler (tested with **g++**)

  > ⚠️ Compatibility with other compilers is not guaranteed.  
  > If issues arise, check `get_lsb_index(U64 bb)` in `include/utils.hpp`, which may rely on compiler-specific intrinsics.

- CMake installed

> ⚠️ While designed with 32-bit compatibility in mind, the engine has not been tested on 32-bit systems.

#### Compatibility concerns

- The engine has been tested on both Linux (x86 CPU architecture) and MacOS (ARM CPU Architecture). However, the engine hasn't been yet tested on Windows. Issues might arise as the multithreading libraries on UNIX like systems differ from Windows ones.

### Build Instructions

Clone the repository:

```bash
git clone https://github.com/EmericBraud/chess26.git
cd chess-26
```

Build the project:
cmake --build .

Run the engine:
./build/chess_26

> Note: a Makefile is present at the root directory. You can simply use `make`then `./chess26``

### Running Tests

The project includes unit tests located in the tests/ directory:

```bash
ctest
```

## 📝 License

This project is licensed under the MIT License.

### Dependencies

This project depends on:

- fathom: a C project developped by jdart1 that helps probing Syzygy tables - link here: https://github.com/jdart1/Fathom (MIT license)
- SFML (if ENABLE_GUI option is set): a C++ GUI library developped by Laurent Gormilla- link here: https://github.com/sfml/sfml (Zlib license)
- Google Tests - link here: https://github.com/google/googletest (BSD-3-Clause license)

### Credits

I want to thank the http://chessprogramming.org community for sharing such precious information.
Thanks to OpenBench creators (link: https://github.com/andygrant/openbench) for sharing such an amazing SPSA tuning tool.
I also greatly used lichess-bot software (link: https://github.com/lichess-bot-devs/lichess-bot) to connect my UCI engine to lichess.org.
Texel (link: https://github.com/peterosterlund2/texel) tuning technic has also helped me greatly improve my HCE function to improve.
Finally, thanks to Stockfish (link: https://github.com/official-stockfish/stockfish), Ethereal (link: https://github.com/AndyGrant/Ethereal) communities for having developped and shared such performant chess engines.

## 👨‍💻 Author

Emeric Braud
https://www.linkedin.com/in/emeric-braud-101239151/
