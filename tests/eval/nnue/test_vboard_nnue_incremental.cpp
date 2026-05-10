#include "gtest/gtest.h"

#include <string>
#include <vector>
#include <iostream>

#include "engine/eval/virtual_board.hpp"

#ifndef NNUE_EVAL
TEST(VBoardNnueIncrementalTest, RequiresNnueBuild)
{
    GTEST_SKIP() << "NNUE_EVAL is not enabled in this test build.";
}
#else

namespace
{
    void expect_incremental_matches_refresh(
        const std::string &fen,
        const std::vector<std::string> &moves_uci)
    {
        VBoard board;
        ASSERT_TRUE(board.load_fen(fen));

        const int baseline_raw = board.get_nnue_eval().evaluate_abs();

        std::vector<Move> played;
        played.reserve(moves_uci.size());

        for (const auto &uci : moves_uci)
        {
            auto parsed = Board::parse_move_uci(uci, board);
            ASSERT_TRUE(parsed.has_value()) << "Invalid move: " << uci;
            const Move move = *parsed;

            ASSERT_TRUE(board.is_move_pseudo_legal(move)) << "Move not pseudo-legal: " << uci;
            ASSERT_TRUE(board.is_move_legal(move)) << "Move not legal: " << uci;

            board.play(move);
            played.push_back(move);

            VBoard refreshed(static_cast<const Board &>(board));
            EXPECT_EQ(board.get_nnue_eval().evaluate_abs(), refreshed.get_nnue_eval().evaluate_abs())
                << "Incremental NNUE mismatch after move " << uci;
        }

        for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        {
            board.unplay(played[i]);

            VBoard refreshed(static_cast<const Board &>(board));
            EXPECT_EQ(board.get_nnue_eval().evaluate_abs(), refreshed.get_nnue_eval().evaluate_abs())
                << "Incremental NNUE mismatch after unplay index " << i;
        }

        EXPECT_EQ(board.get_nnue_eval().evaluate_abs(), baseline_raw)
            << "NNUE raw score did not return to baseline after full unplay";
    }
}

TEST(VBoardNnueIncrementalTest, CaptureSequence)
{
    expect_incremental_matches_refresh(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {"e2e4", "d7d5", "e4d5"});
}

TEST(VBoardNnueIncrementalTest, EnPassantSequence)
{
    expect_incremental_matches_refresh(
        "rnbqkbnr/pppppppp/8/3Pp3/8/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        {"d5e6"});
}

TEST(VBoardNnueIncrementalTest, PromotionSequence)
{
    expect_incremental_matches_refresh(
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        {"a7a8q"});
}

TEST(VBoardNnueIncrementalTest, CastlingAndKingMovesSequence)
{
    expect_incremental_matches_refresh(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        {"e1g1", "e8c8", "g1g2", "c8d7"});
}

TEST(VBoardNnueIncrementalTest, RawNnueOutputChangesAndRestores)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    const int raw_before = board.get_nnue_eval().evaluate_abs();

    std::vector<Move> played;
    for (const char *uci : {"e2e4", "d7d5", "e4d5"})
    {
        auto parsed = Board::parse_move_uci(uci, board);
        ASSERT_TRUE(parsed.has_value()) << "Invalid move: " << uci;
        const Move move = *parsed;
        ASSERT_TRUE(board.is_move_pseudo_legal(move)) << "Not pseudo-legal: " << uci;
        ASSERT_TRUE(board.is_move_legal(move)) << "Not legal: " << uci;
        board.play(move);
        played.push_back(move);
    }

    const int raw_after_capture = board.get_nnue_eval().evaluate_abs();

    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);

    const int raw_restored = board.get_nnue_eval().evaluate_abs();

    std::cout
        << "NNUE raw start=" << raw_before
        << " after_e4xd5=" << raw_after_capture
        << " restored=" << raw_restored
        << std::endl;

    EXPECT_EQ(raw_before, raw_restored);
}

#endif
