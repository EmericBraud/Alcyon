#include "gtest/gtest.h"

#include <bit>
#include <string>
#include <vector>

#include "engine/eval/virtual_board.hpp"

// Mirrors tests/eval/nnue/test_vboard_nnue_incremental.cpp (v1's incremental-
// consistency test), but for the v2 (Full_Threats + HalfKAv2_hm^) accumulator
// wired into VBoard's play/unplay hooks (see virtual_board.hpp). Confirms the
// incrementally-updated accumulator matches a from-scratch initialize() at
// every step of a move sequence, for quiet moves, captures, king moves,
// castling, en passant and promotions.
#ifndef NNUE_EVAL_V2
TEST(VBoardNnueV2IncrementalTest, RequiresNnueV2Build)
{
    GTEST_SKIP() << "NNUE_EVAL_V2 is not enabled in this test build.";
}
#else

namespace
{
    int piece_count(const VBoard &board)
    {
        return std::popcount(board.get_occupancy<NO_COLOR>());
    }

    void expect_incremental_matches_refresh(
        const std::string &fen,
        const std::vector<std::string> &moves_uci)
    {
        VBoard board;
        ASSERT_TRUE(board.load_fen(fen));

        const int baseline_raw = board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board));

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
            EXPECT_EQ(board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board)),
                      refreshed.get_nnue_eval_v2().evaluate_abs(refreshed.get_side_to_move(), piece_count(refreshed)))
                << "Incremental NNUE v2 mismatch after move " << uci;
        }

        for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        {
            board.unplay(played[i]);

            VBoard refreshed(static_cast<const Board &>(board));
            EXPECT_EQ(board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board)),
                      refreshed.get_nnue_eval_v2().evaluate_abs(refreshed.get_side_to_move(), piece_count(refreshed)))
                << "Incremental NNUE v2 mismatch after unplay index " << i;
        }

        EXPECT_EQ(board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board)), baseline_raw)
            << "NNUE v2 raw score did not return to baseline after full unplay";
    }
}

TEST(VBoardNnueV2IncrementalTest, QuietMoveSequence)
{
    expect_incremental_matches_refresh(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {"g1f3", "g8f6", "b1c3", "b8c6"});
}

TEST(VBoardNnueV2IncrementalTest, CaptureSequence)
{
    expect_incremental_matches_refresh(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {"e2e4", "d7d5", "e4d5"});
}

TEST(VBoardNnueV2IncrementalTest, TacticalCaptureSequence)
{
    // Position with several sliders on shared lines, to exercise the
    // scoped-recompute path's "recompute every slider" behavior.
    expect_incremental_matches_refresh(
        "r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1",
        {"d4c5", "d6c5", "f3e4", "d7e6"});
}

TEST(VBoardNnueV2IncrementalTest, EnPassantSequence)
{
    expect_incremental_matches_refresh(
        "rnbqkbnr/pppppppp/8/3Pp3/8/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
        {"d5e6"});
}

TEST(VBoardNnueV2IncrementalTest, PromotionSequence)
{
    expect_incremental_matches_refresh(
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        {"a7a8q"});
}

TEST(VBoardNnueV2IncrementalTest, PromotionWithCaptureSequence)
{
    expect_incremental_matches_refresh(
        "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        {"a7b8q"});
}

TEST(VBoardNnueV2IncrementalTest, CastlingAndKingMovesSequence)
{
    expect_incremental_matches_refresh(
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        {"e1g1", "e8c8", "g1g2", "c8d7"});
}

TEST(VBoardNnueV2IncrementalTest, RawNnueOutputChangesAndRestores)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"));

    const int raw_before = board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board));

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

    for (int i = static_cast<int>(played.size()) - 1; i >= 0; --i)
        board.unplay(played[i]);

    const int raw_restored = board.get_nnue_eval_v2().evaluate_abs(board.get_side_to_move(), piece_count(board));

    EXPECT_EQ(raw_before, raw_restored);
}

#endif
