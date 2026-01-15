#include "core/move/generator/move_generator.hpp"
#include "gtest/gtest.h"

class MoveLogicTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        MoveGen::initialize_bitboard_tables();
    }
    void SetUp() override
    {
    }
};

TEST_F(MoveLogicTest, SlidingAttacks)
{
    U64 mask = generate_sliding_attack(Square::d3, 0ULL, true);
    ASSERT_EQ(mask, 0b100000001000000010000000100000001000111101110000100000001000);
    mask = generate_sliding_attack(Square::d3, 0ULL, false);
    ASSERT_EQ(mask, 0b10000000010000010010001000010100000000000001010000100010);
    mask = generate_sliding_attack(Square::d5, 0b1000000000000000001000000000000000000000100000000000, true);
    ASSERT_EQ(mask, 0b1000000010001111011000001000000010000000100000000000);
}

TEST_F(MoveLogicTest, RookMask)
{
    U64 mask = MoveGen::RookMasks[Square::d5];
    ASSERT_EQ(mask, 0b1000000010000111011000001000000010000000100000000000);
}