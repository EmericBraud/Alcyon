#include "gtest/gtest.h"

#include <bit>
#include <random>
#include <string>
#include <vector>

#include "common/constants.hpp"
#include "engine/config/nnue.hpp"
#include "engine/eval/virtual_board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "core/move/move_list.hpp"

// Coverage for the from-scratch accumulator rebuild path (see
// NnueEval::materialize(board)): when a full eval has to catch up a long
// backlog of buffered plies (>= engine_constants::nnue::AccRebuildMinPlies,
// or the lower AccRebuildKingMinPlies when the backlog holds a king move),
// the L1 accumulator is rebuilt from the current board under a single
// depth-tagged snapshot instead of replaying every per-ply diff. These tests
// pin the behavior contract around that jump: the eval must be identical to
// a fresh recompute, unwinding back *through* a jump must land on a state
// from which skipped plies can still re-materialize, and the independently
// materializing PSQT half must never be disturbed by an L1 jump.
#ifndef NNUE_EVAL
TEST(VBoardNnueRebuildTest, RequiresNnueBuild)
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
        return board.get_nnue_eval().evaluate_abs(board);
    }

    int psqt_of(const VBoard &board)
    {
        return board.get_nnue_eval().evaluate_psqt_abs(board.get_side_to_move(), piece_count(board));
    }

    // Full-recompute references: a fresh VBoard built off the raw Board
    // state has no pending backlog at all, so its eval can't involve any
    // replay/rebuild decision.
    int reference_eval(const VBoard &board)
    {
        VBoard refreshed(static_cast<const Board &>(board));
        return eval_of(refreshed);
    }

    int reference_psqt(const VBoard &board)
    {
        VBoard refreshed(static_cast<const Board &>(board));
        return psqt_of(refreshed);
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

    // 16 plies from the starting position with NO king move anywhere
    // (Najdorf/English-attack line): long enough to exceed
    // AccRebuildMinPlies while keeping every buffered diff purely
    // incremental (no pd.refresh in the backlog).
    const std::vector<std::string> long_quiet_line = {
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6",
        "b1c3", "a7a6", "f2f3", "e7e5", "d4b3", "c8e6", "c1e3", "b8d7"};

    // 7 plies ending with a castle: above AccRebuildKingMinPlies but below
    // AccRebuildMinPlies, so this backlog only crosses the rebuild cutover
    // through the king-move rule.
    const std::vector<std::string> short_castle_line = {
        "e2e4", "e7e5", "g1f3", "g8f6", "f1c4", "f8c5", "e1g1"};

    // 16 plies with BOTH sides castling plus a king recapture pattern
    // (Berlin): several pd.refresh plies in one backlog.
    const std::vector<std::string> berlin_line = {
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1", "f6e4",
        "f1e1", "e4d6", "f3e5", "f8e7", "b5f1", "c6e5", "e1e5", "e8g8"};

    static_assert(engine_constants::nnue::AccRebuildMinPlies <= 16,
                  "the scripted 16-ply lines must exceed the rebuild threshold");
    static_assert(engine_constants::nnue::AccRebuildKingMinPlies <= 7,
                  "the scripted 7-ply castle line must reach the king-move threshold");
}

// A single eval after a backlog longer than AccRebuildMinPlies: the rebuild
// jump must produce exactly the same eval as a fresh recompute.
TEST(VBoardNnueRebuildTest, LongQuietBacklogThenEval)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<Move> played;
    play_all(board, long_quiet_line, played);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after a 16-ply un-evaluated quiet backlog";
}

// King-move backlogs cross the cutover earlier (AccRebuildKingMinPlies).
TEST(VBoardNnueRebuildTest, ShortBacklogWithCastle)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<Move> played;
    play_all(board, short_castle_line, played);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after a 7-ply backlog ending with O-O";
}

// Several pd.refresh plies (both castlings + a rook-takes recapture on the
// king file) inside one long backlog.
TEST(VBoardNnueRebuildTest, LongBacklogWithMultipleKingMoves)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<Move> played;
    play_all(board, berlin_line, played);

    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval after the 16-ply Berlin backlog";
}

// Unwinding back through a rebuild jump: the single tagged snapshot must
// restore the pre-backlog state, and every skipped ply's still-buffered diff
// must be able to re-materialize when an eval is requested mid-unwind.
TEST(VBoardNnueRebuildTest, UnwindThroughJumpThenReEval)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    const int baseline = eval_of(board);

    std::vector<Move> played;
    play_all(board, long_quiet_line, played);

    ASSERT_EQ(eval_of(board), reference_eval(board));

    // Unplay half the line, then re-eval: the plies below the jump were
    // never individually materialized, so this forces a second catch-up
    // (replay or a fresh jump, depending on the remaining backlog).
    for (int i = 0; i < 8; ++i)
    {
        board.unplay(played.back());
        played.pop_back();
    }
    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Re-eval halfway down through the jump";

    // Then evaluate at every remaining ply on the way down.
    while (!played.empty())
    {
        board.unplay(played.back());
        played.pop_back();
        EXPECT_EQ(eval_of(board), reference_eval(board)) << "Eval during final unwind, " << played.size() << " plies left";
    }
    EXPECT_EQ(eval_of(board), baseline) << "Back to the starting position";
}

// Sibling pattern (the search's common shape): eval a leaf reached through a
// jump, pop one ply, branch to a different move, eval again.
TEST(VBoardNnueRebuildTest, SiblingEvalAfterJump)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<Move> played;
    play_all(board, long_quiet_line, played);
    ASSERT_EQ(eval_of(board), reference_eval(board));

    board.unplay(played.back());
    played.pop_back();

    const Move sibling = parse_checked("f8e7", board);
    board.play(sibling);
    EXPECT_EQ(eval_of(board), reference_eval(board)) << "Sibling leaf eval right after a jump+pop";
}

// The PSQT half materializes independently (its own counter and per-ply
// snapshot stack): keeping it fully caught up ply by ply while the L1 half
// jumps -- and vice versa, leaving it behind -- must never desync either.
TEST(VBoardNnueRebuildTest, PsqtInterleavedWithJump)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    VBoard psqt_stale(board); // same line, but no psqt eval along the way

    std::vector<Move> played, played_stale;
    for (const auto &uci : long_quiet_line)
    {
        const Move move = parse_checked(uci, board);
        board.play(move);
        played.push_back(move);
        EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "psqt after " << uci;

        const Move move_stale = parse_checked(uci, psqt_stale);
        psqt_stale.play(move_stale);
        played_stale.push_back(move_stale);
    }

    // board: psqt caught up ply-by-ply, L1 jumps now.
    EXPECT_EQ(eval_of(board), reference_eval(board)) << "L1 jump over a psqt-materialized backlog";
    // psqt_stale: both halves catch up at once (L1 jumps, psqt replays).
    EXPECT_EQ(eval_of(psqt_stale), reference_eval(psqt_stale)) << "L1 jump + psqt replay together";
    EXPECT_EQ(psqt_of(psqt_stale), reference_psqt(psqt_stale));

    // Unwind both through the jump with mixed eval kinds.
    for (int i = 0; i < 10; ++i)
    {
        board.unplay(played.back());
        played.pop_back();
        psqt_stale.unplay(played_stale.back());
        played_stale.pop_back();
    }
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
    EXPECT_EQ(eval_of(board), reference_eval(board));
    EXPECT_EQ(eval_of(psqt_stale), reference_eval(psqt_stale));
    EXPECT_EQ(psqt_of(psqt_stale), reference_psqt(psqt_stale));
}

// Copying a VBoard whose source sits on top of an un-materialized long
// backlog: the copy materializes the source (possibly via a jump) and must
// evaluate identically, including after diverging lines on both boards.
TEST(VBoardNnueRebuildTest, CopyAcrossPendingBacklog)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    std::vector<Move> played;
    play_all(board, long_quiet_line, played);

    VBoard copy(board);
    EXPECT_EQ(eval_of(copy), reference_eval(copy)) << "Copy constructed over a 16-ply backlog";

    const Move original_next = parse_checked("f1e2", board);
    board.play(original_next);
    const Move copy_next = parse_checked("d1d2", copy);
    copy.play(copy_next);

    EXPECT_EQ(eval_of(board), reference_eval(board));
    EXPECT_EQ(eval_of(copy), reference_eval(copy));

    copy.unplay(copy_next);
    EXPECT_EQ(eval_of(copy), reference_eval(copy)) << "Copy after unplaying its own branch";
}

// Randomized cross-check: long random games with *sparse* full evals (so
// backlogs of every length around both thresholds occur), sparse psqt
// evals, and random partial unwinds -- every eval compared against a fresh
// from-scratch recompute.
TEST(VBoardNnueRebuildTest, RandomWalkSparseEvals)
{
    const std::vector<std::string> fens = {
        std::string(constants::FenInitPos),
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"};

    std::mt19937 rng(20260726);

    for (const auto &fen : fens)
    {
        VBoard board;
        ASSERT_TRUE(board.load_fen(fen));
        std::vector<Move> played;

        for (int step = 0; step < 400; ++step)
        {
            const int roll = static_cast<int>(rng() % 100);

            if (roll < 60 || played.empty())
            {
                MoveList list;
                MoveGen::generate_legal_moves(board, list);
                if (list.count == 0)
                    break;
                const Move move = list[static_cast<int>(rng() % list.count)];
                board.play(move);
                played.push_back(move);
            }
            else if (roll < 75)
            {
                // Partial unwind of a random depth.
                int to_unwind = 1 + static_cast<int>(rng() % played.size());
                while (to_unwind-- > 0)
                {
                    board.unplay(played.back());
                    played.pop_back();
                }
            }
            else if (roll < 90)
            {
                ASSERT_EQ(psqt_of(board), reference_psqt(board)) << fen << " step " << step;
            }
            else
            {
                ASSERT_EQ(eval_of(board), reference_eval(board)) << fen << " step " << step;
            }
        }

        while (!played.empty())
        {
            board.unplay(played.back());
            played.pop_back();
        }
        ASSERT_EQ(eval_of(board), reference_eval(board)) << fen << " after full unwind";
        ASSERT_EQ(psqt_of(board), reference_psqt(board)) << fen << " after full unwind";
    }
}

#endif
