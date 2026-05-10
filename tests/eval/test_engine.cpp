#include "engine/engine_manager.hpp"
#include "gtest/gtest.h"
#include "engine/eval/virtual_board.hpp"

class EngineTest : public ::testing::Test
{
protected:
    Board b;

    void SetUp() override
    {
    }
};
TEST_F(EngineTest, EngineCanCastle)
{
    VBoard b;
    b.load_fen("2q1k3/8/8/8/8/3PPPPP/3PPPPP/4K2R w K - 0 1");
    EngineManager e{b};
    e.start_search(1000);
    e.wait();
    b.play(e.get_root_best_move());
    ASSERT_EQ(b.get_piece_bitboard(WHITE, KING), 0x40);
}

TEST_F(EngineTest, PinTest)
{
    VBoard b;
    b.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1");
    EngineManager e{b};
    e.start_search(500);
    e.wait();
    Move last_move = e.get_root_best_move();
    ASSERT_EQ(last_move, Move(Square::h2, Square::h3, PAWN));
}

TEST_F(EngineTest, FixedDepthScoreChangesWithMaterial)
{
    VBoard equalish;
    ASSERT_TRUE(equalish.load_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"));
    const int equal_raw = equalish.get_nnue_eval().evaluate_abs();
    EngineManager equal_manager{equalish};
    const auto equal_result = equal_manager.run_benchmark_fixed_depth(equalish, 1);

    VBoard queen_up;
    ASSERT_TRUE(queen_up.load_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K2Q w - - 0 1"));
    const int queen_raw = queen_up.get_nnue_eval().evaluate_abs();
    EngineManager queen_manager{queen_up};
    const auto queen_result = queen_manager.run_benchmark_fixed_depth(queen_up, 1);

    EXPECT_NE(equal_raw, queen_raw);
    EXPECT_NE(equal_result.score_cp, queen_result.score_cp);
}

TEST_F(EngineTest, QueenMoveChangesRawNnue)
{
    VBoard board;
    ASSERT_TRUE(board.load_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K2Q w - - 0 1"));

    const int before = board.get_nnue_eval().evaluate_abs();
    auto parsed = Board::parse_move_uci("h1h2", board);
    ASSERT_TRUE(parsed.has_value());
    ASSERT_TRUE(board.is_move_pseudo_legal(*parsed));
    ASSERT_TRUE(board.is_move_legal(*parsed));

    board.play(*parsed);
    const int after = board.get_nnue_eval().evaluate_abs();

    EXPECT_NE(before, after);
}