#include "gtest/gtest.h"

#include <bit>
#include <random>
#include <string>
#include <vector>

#include "engine/eval/virtual_board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "core/move/move_list.hpp"

// Coverage for *sparse* evaluation patterns: unlike
// test_vboard_nnue_incremental.cpp (which evaluates after every single
// play/unplay), these tests deliberately interleave plays, unplays, copies
// and evaluations in irregular ways. This is the behavior contract a
// lazy-apply accumulator (diffs buffered per ply, only materialized when an
// eval actually happens) must preserve: an eval may be requested after
// several un-evaluated plays, after unplaying moves that were never
// evaluated, or after unwinding below the last materialized ply.
#ifndef NNUE_EVAL
TEST(VBoardNnueLazyTest, RequiresNnueBuild)
{
    GTEST_SKIP() << "NNUE_EVAL is not enabled in this test build.";
}
#else

namespace
{
    int piece_count(const VBoard &board)
    {
        return std::popcount(board.get_occupancy<NO_COLOR>());
    }

    int eval_of(const VBoard &board)
    {
        return board.get_nnue_eval().evaluate_abs(board.get_side_to_move(), piece_count(board));
    }

    // Full-recompute reference: builds a fresh VBoard (fresh accumulator,
    // initialized from scratch off the raw Board state) and evaluates it.
    int reference_eval(const VBoard &board)
    {
        VBoard refreshed(static_cast<const Board &>(board));
        return eval_of(refreshed);
    }

    Move parse_checked(const std::string &uci, VBoard &board)
    {
        auto parsed = Board::parse_move_uci(uci, board);
        EXPECT_TRUE(parsed.has_value()) << "Invalid move: " << uci;
        EXPECT_TRUE(board.is_move_pseudo_legal(*parsed)) << "Not pseudo-legal: " << uci;
        EXPECT_TRUE(board.is_move_legal(*parsed)) << "Not legal: " << uci;
        return *parsed;
    }

    void play_all(VBoard &board, const std::vector<std::string> &moves_uci, std::vector<Move> &played)
    {
        for (const auto &uci : moves_uci)
        {
            const Move move = parse_checked(uci, board);
            board.play(move);
            played.push_back(move);
        }
    }
}

// Several plays with NO intermediate eval, a single eval at the end: after a
// lazy-apply refactor this materializes a whole stack of buffered diffs in
// one go.
TEST(VBoardNnueLazyTest, EvalOnlyAtEndOfSequence)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"f3f6", "g7f6", "e5g6", "f7g6", "e1g1", "e8g8"}, played);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after 6 un-evaluated plays";
}

// Plays followed by unplays with no eval in between: buffered diffs must be
// discarded (not applied) when their ply is unwound before any eval.
TEST(VBoardNnueLazyTest, PlayUnplayWithoutEvalThenEval)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    const int baseline = eval_of(board);

    std::vector<Move> played;
    play_all(board, {"e2e4", "d7d5", "e4d5", "d8d5"}, played);

    // Unwind everything without ever evaluating.
    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);

    EXPECT_EQ(eval_of(board), baseline) << "Eval after un-evaluated play/unplay round trip";
    EXPECT_EQ(eval_of(board), reference_eval(board));
}

// Unwind PART of an un-evaluated line, then branch to a different move and
// only then evaluate: mixes discarded diffs with still-pending ones.
TEST(VBoardNnueLazyTest, PartialUnwindThenBranchThenEval)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1"));

    std::vector<Move> played;
    play_all(board, {"d4c5", "d6c5", "f3e4"}, played);

    // Unwind the last two plies without evaluating.
    board.unplay(played[2]);
    board.unplay(played[1]);
    played.resize(1);

    // Branch differently, then evaluate for the first time.
    const Move branch = parse_checked("e4c3", board);
    board.play(branch);
    played.push_back(branch);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after partial unwind + branch";

    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);
    EXPECT_EQ(eval_of(board), reference_eval(board));
}

// Evaluate mid-line (materializing up to that ply), go deeper without
// evaluating, then unwind below the evaluated ply and evaluate again:
// exercises the boundary between already-applied and still-buffered plies.
TEST(VBoardNnueLazyTest, EvalMidlineThenUnwindBelowIt)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"f3f6", "g7f6"}, played);
    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval at ply 2";

    play_all(board, {"e5g6", "f7g6"}, played);

    // Unwind everything -- the last two plies were never evaluated, the
    // first two were.
    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after unwinding below a materialized ply";
}

// King moves and castling with evals deferred several plies: a lazy refactor
// must keep the deferred "full refresh of the mover's perspective" coherent
// with later buffered diffs stacked on top of it.
TEST(VBoardNnueLazyTest, DeferredEvalAcrossKingMoves)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"e1g1", "e8c8", "g1g2", "c8d7", "a1e1"}, played);
    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after deferred king-move sequence";

    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);
    EXPECT_EQ(eval_of(board), reference_eval(board));
}

// Copying a VBoard (copy-ctor and assignment) while un-evaluated plays are
// outstanding: both the copy and the original must evaluate correctly
// afterwards, independently.
TEST(VBoardNnueLazyTest, CopyWhilePendingDiffsOutstanding)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"e2e4", "d7d5", "e4d5"}, played);

    // No eval has happened yet on `board`.
    VBoard copy_constructed(board);
    EXPECT_EQ(eval_of(copy_constructed), reference_eval(board)) << "Copy-constructed while pending";

    VBoard assigned;
    ASSERT_TRUE(assigned.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    assigned = board;
    EXPECT_EQ(eval_of(assigned), reference_eval(board)) << "Copy-assigned while pending";

    // The original must still evaluate and unwind correctly after being
    // copied from.
    EXPECT_EQ(eval_of(board), reference_eval(board));
    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);
    EXPECT_EQ(eval_of(board), reference_eval(board));

    // The copy can keep playing on its own line.
    const Move copy_move = parse_checked("d8d5", copy_constructed);
    copy_constructed.play(copy_move);
    EXPECT_EQ(eval_of(copy_constructed), reference_eval(copy_constructed));
}

// Seeded random walks: random legal games with sparse (probabilistic) evals
// and occasional random partial unwinds, checked against a full recompute at
// every eval point. Catches interleavings the hand-written cases above miss.
TEST(VBoardNnueLazyTest, RandomWalkSparseEvals)
{
    MoveGen::initialize_bitboard_tables();

    const std::vector<std::string> start_fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };

    std::mt19937 rng(20260725);

    for (const auto &fen : start_fens)
    {
        VBoard board;
        ASSERT_TRUE(board.load_fen(fen));
        const int baseline = eval_of(board);

        std::vector<Move> played;
        int evals_done = 0;

        for (int step = 0; step < 60; ++step)
        {
            MoveList list;
            if (board.get_side_to_move() == WHITE)
                MoveGen::generate_legal_moves<WHITE>(board, list);
            else
                MoveGen::generate_legal_moves<BLACK>(board, list);
            if (list.count == 0)
                break;

            const Move move = list[std::uniform_int_distribution<int>(0, list.count - 1)(rng)];
            board.play(move);
            played.push_back(move);

            // ~20% of plies get an eval; the rest stay un-evaluated.
            if (std::uniform_int_distribution<int>(0, 4)(rng) == 0)
            {
                ++evals_done;
                ASSERT_EQ(eval_of(board), reference_eval(board))
                    << "Random-walk mismatch, fen=" << fen << " step=" << step;
            }

            // ~15% of plies: unwind 1..3 plies (possibly through evaluated
            // and un-evaluated plies alike) and keep walking from there.
            if (!played.empty() && std::uniform_int_distribution<int>(0, 6)(rng) == 0)
            {
                const int max_back = std::min<int>(3, static_cast<int>(played.size()));
                const int back = std::uniform_int_distribution<int>(1, max_back)(rng);
                for (int i = 0; i < back; ++i)
                {
                    board.unplay(played.back());
                    played.pop_back();
                }
            }
        }

        ASSERT_GT(evals_done, 0) << "Random walk never evaluated, fen=" << fen;

        for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
            board.unplay(played[i]);
        EXPECT_EQ(eval_of(board), baseline) << "fen=" << fen;
        EXPECT_EQ(eval_of(board), reference_eval(board)) << "fen=" << fen;
    }
}

#endif
