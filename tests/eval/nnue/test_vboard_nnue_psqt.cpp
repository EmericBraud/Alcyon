#include "gtest/gtest.h"

#include <bit>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "engine/eval/pos_eval.hpp"
#include "engine/eval/virtual_board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "core/move/move_list.hpp"

// Coverage for the NNUE PSQT-only fast eval (evaluate_psqt_abs), which
// replaces the HCE EvalState-based lazy_eval_relative in NNUE builds: the
// PSQT accumulator half of the network is materialized independently of the
// (much heavier) L1 feature-transformer accumulator, so a pruning heuristic
// can ask for a cheap material/PSQT estimate at plies where no full eval
// ever happens. These tests interleave psqt-only evals, full evals, unplays
// and copies in irregular orders: whatever materialization bookkeeping the
// implementation uses, a psqt eval must always equal the value a
// from-scratch rebuild of the same position would give.
#ifndef NNUE_EVAL
TEST(VBoardNnuePsqtTest, RequiresNnueBuild)
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

    int psqt_of(const VBoard &board)
    {
        return board.get_nnue_eval().evaluate_psqt_abs(board.get_side_to_move(), piece_count(board));
    }

    int eval_of(const VBoard &board)
    {
        return board.get_nnue_eval().evaluate_abs(board);
    }

    // Full-recompute reference: builds a fresh VBoard (fresh accumulators,
    // initialized from scratch off the raw Board state).
    int reference_psqt(const VBoard &board)
    {
        VBoard refreshed(static_cast<const Board &>(board));
        return psqt_of(refreshed);
    }

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

    void unwind_all(VBoard &board, std::vector<Move> &played)
    {
        for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
            board.unplay(played[i]);
        played.clear();
    }
}

// The PSQT head is the material/PSQT part of the network: it must be small
// on the symmetric starting position and strongly favor the side to move
// when it is up a queen. This pins the output scaling (a factor-of-2 or
// sign mistake in the quantized psqt_diff -> centipawn conversion would
// blow straight through these bounds).
TEST(VBoardNnuePsqtTest, ScaleAndSignSanity)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    EXPECT_LT(std::abs(psqt_of(board)), 60) << "Startpos PSQT should be near zero";

    // White is up a full queen, white to move.
    ASSERT_TRUE(board.load_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));
    const int up_queen = psqt_of(board);
    EXPECT_GT(up_queen, 300) << "PSQT should see a queen of material";

    // Same position, black to move: side-to-move-relative sign must flip.
    ASSERT_TRUE(board.load_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"));
    EXPECT_LT(psqt_of(board), -300) << "PSQT must be side-to-move relative";
}

// Several plays with NO eval of any kind, then a psqt-only eval: the psqt
// half must be materializable on its own, without a full eval ever running.
TEST(VBoardNnuePsqtTest, PsqtOnlyAtEndOfSequence)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"f3f6", "g7f6", "e5g6", "f7g6", "e1g1", "e8g8"}, played);

    EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "PSQT after 6 un-evaluated plays";

    // A full eval afterwards must still be coherent (the psqt-only
    // materialization must not desync the full pipeline).
    EXPECT_EQ(eval_of(board), reference_eval(board));
}

// Plays followed by unplays with only psqt evals in between: buffered diffs
// whose psqt half was applied (but not the accumulator half) must roll back
// correctly on unplay.
TEST(VBoardNnuePsqtTest, PsqtEvalThenUnwind)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    const int baseline_psqt = psqt_of(board);
    const int baseline_eval = eval_of(board);

    std::vector<Move> played;
    play_all(board, {"e2e4", "d7d5", "e4d5", "d8d5"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));

    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), baseline_psqt) << "PSQT after psqt-evaluated round trip";
    EXPECT_EQ(eval_of(board), baseline_eval);
}

// Captures, promotions and en passant all change material: exactly what the
// PSQT head exists to track. Evals are deferred across the whole sequence.
TEST(VBoardNnuePsqtTest, MaterialChangesDeferred)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/pP1p1ppp/8/2pP4/8/8/P1P1PPPP/R3K2R w KQkq c6 0 2"));

    std::vector<Move> played;
    // En passant, then an underpromotion capture on a8.
    play_all(board, {"d5c6", "d7c6", "b7a8q"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "PSQT after EP + promotion capture";
    EXPECT_EQ(eval_of(board), reference_eval(board));

    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
}

// King moves and castling defer a full refresh of the mover's perspective;
// the psqt half of that refresh must be applicable independently.
TEST(VBoardNnuePsqtTest, PsqtAcrossKingMoves)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"e1g1", "e8c8", "g1g2", "c8d7", "a1e1"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "PSQT after deferred king-move sequence";
    EXPECT_EQ(eval_of(board), reference_eval(board));

    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
}

// Interleave psqt-only and full evals at different plies, in both orders:
// whatever independent bookkeeping tracks each half's materialized depth,
// they must never desync (psqt materialized above the accumulator, then a
// full eval below; and the converse).
TEST(VBoardNnuePsqtTest, InterleavedPsqtAndFullEvals)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"));

    std::vector<Move> played;

    // psqt first, full eval deeper.
    play_all(board, {"f3f6"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
    play_all(board, {"g7f6", "e5g6"}, played);
    EXPECT_EQ(eval_of(board), reference_eval(board));
    EXPECT_EQ(psqt_of(board), reference_psqt(board));

    // Go deeper: full eval first, psqt after.
    play_all(board, {"f7g6", "e1g1"}, played);
    EXPECT_EQ(eval_of(board), reference_eval(board));
    play_all(board, {"e8g8"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));

    // Unwind through plies with mixed materialization states.
    board.unplay(played.back());
    played.pop_back();
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
    EXPECT_EQ(eval_of(board), reference_eval(board));

    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
    EXPECT_EQ(eval_of(board), reference_eval(board));
}

// Unwind part of a line where only psqt evals happened, branch elsewhere,
// then evaluate fully: mixes discarded psqt-applied diffs with pending ones.
TEST(VBoardNnuePsqtTest, PartialUnwindThenBranch)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1"));

    std::vector<Move> played;
    play_all(board, {"d4c5", "d6c5", "f3e4"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));

    board.unplay(played[2]);
    board.unplay(played[1]);
    played.resize(1);

    const Move branch = parse_checked("e4c3", board);
    board.play(branch);
    played.push_back(branch);

    EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "PSQT after partial unwind + branch";
    EXPECT_EQ(eval_of(board), reference_eval(board));

    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
}

// Copies taken while some plies are psqt-materialized but not fully
// materialized: both the copy and the original must stay coherent.
TEST(VBoardNnuePsqtTest, CopyWithMixedMaterialization)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    std::vector<Move> played;
    play_all(board, {"e2e4", "d7d5", "e4d5"}, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board)); // psqt-materialized, acc still pending

    VBoard copy_constructed(board);
    EXPECT_EQ(psqt_of(copy_constructed), reference_psqt(board));
    EXPECT_EQ(eval_of(copy_constructed), reference_eval(board));

    VBoard assigned;
    ASSERT_TRUE(assigned.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"));
    assigned = board;
    EXPECT_EQ(psqt_of(assigned), reference_psqt(board));

    EXPECT_EQ(eval_of(board), reference_eval(board));
    unwind_all(board, played);
    EXPECT_EQ(psqt_of(board), reference_psqt(board));
}

// The NNUE-mode lazy_eval_relative must be wired to the PSQT head (white
// perspective matches evaluate_psqt_abs up to the side-to-move sign flip).
TEST(VBoardNnuePsqtTest, LazyEvalRelativeUsesPsqt)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"));

    const int psqt_stm = psqt_of(board); // black to move, black-relative
    EXPECT_EQ(Eval::lazy_eval_relative<BLACK>(board), psqt_stm);
    EXPECT_EQ(Eval::lazy_eval_relative<WHITE>(board), -psqt_stm);
}

// Seeded random walks with three interleaved event kinds (psqt evals, full
// evals, partial unwinds), each checked against a from-scratch rebuild.
TEST(VBoardNnuePsqtTest, RandomWalkMixedEvals)
{
    MoveGen::initialize_bitboard_tables();

    const std::vector<std::string> start_fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    };

    std::mt19937 rng(20260726);

    for (const auto &fen : start_fens)
    {
        VBoard board;
        ASSERT_TRUE(board.load_fen(fen));
        const int baseline_psqt = psqt_of(board);

        std::vector<Move> played;
        int psqt_evals_done = 0;

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

            const int roll = std::uniform_int_distribution<int>(0, 9)(rng);
            if (roll < 3) // ~30%: psqt-only eval
            {
                ++psqt_evals_done;
                ASSERT_EQ(psqt_of(board), reference_psqt(board))
                    << "Random-walk psqt mismatch, fen=" << fen << " step=" << step;
            }
            else if (roll < 5) // ~20%: full eval
            {
                ASSERT_EQ(eval_of(board), reference_eval(board))
                    << "Random-walk full-eval mismatch, fen=" << fen << " step=" << step;
            }

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

        ASSERT_GT(psqt_evals_done, 0) << "Random walk never psqt-evaluated, fen=" << fen;

        unwind_all(board, played);
        EXPECT_EQ(psqt_of(board), baseline_psqt) << "fen=" << fen;
        EXPECT_EQ(psqt_of(board), reference_psqt(board)) << "fen=" << fen;
        EXPECT_EQ(eval_of(board), reference_eval(board)) << "fen=" << fen;
    }
}

#endif
