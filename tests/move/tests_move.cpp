#include "core/move/generator/move_generator.hpp"
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
