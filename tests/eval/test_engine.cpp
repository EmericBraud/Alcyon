#include "engine/engine_manager.hpp"
#include "gtest/gtest.h"

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
    Board b;
    b.load_fen("2q1k3/8/8/8/8/3PPPPP/3PPPPP/4K2R w K - 0 1");
    EngineManager e{b};
    e.start_search(1000);
    e.wait();
    b.play(e.get_root_best_move());
    ASSERT_EQ(b.get_piece_bitboard(WHITE, KING), 0x40);
}

TEST_F(EngineTest, PinTest)
{
    Board b;
    b.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1");
    EngineManager e{b};
    e.start_search(500);
    e.wait();
    Move last_move = e.get_root_best_move();
    ASSERT_EQ(last_move, Move(Square::h2, Square::h3, PAWN));
}