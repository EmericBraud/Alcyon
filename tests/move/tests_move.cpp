#include "move_generator.hpp"
#include "gtest/gtest.h"

class MoveTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(MoveTest, MoveDescriptor)
{
    Move m{Square::a2, Square::c3, KNIGHT};

    ASSERT_EQ(m.get_from_piece(), KNIGHT);
    ASSERT_EQ(m.get_from_sq(), static_cast<uint32_t>(Square::a2));
    ASSERT_EQ(m.get_to_sq(), static_cast<uint32_t>(Square::c3));
    ASSERT_EQ(m.is_castling(), false);
    ASSERT_EQ(m.is_promotion(), false);
}

TEST_F(MoveTest, EnPassantFile)
{
    Move m{Square::a2, Square::c3, KNIGHT};
    m.set_prev_en_passant(Square::e6 % 8);
    ASSERT_EQ(m.get_prev_en_passant(WHITE), Square::e6);
}