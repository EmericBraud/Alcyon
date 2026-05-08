#include "gtest/gtest.h"
#include "engine/eval/nnue/accumulator_layer.hpp"

#include <cstdint>
#include "core/piece/color.hpp"

class AccumulatorLayerTest : public ::testing::Test
{
protected:
    AccumulatorLayer<4, 2> acc;

    AccumulatorLayerTest()
        : acc(
              std::array<std::int16_t, 2>{{2, -2}},
              std::array<std::array<std::int8_t, 2>, 4>{{{{5, -5}},
                                                         {{2, -2}},
                                                         {{0, 0}},
                                                         {{0, 0}}}})
    {
    }
    void SetUp() override
    {
        acc.reset();
    }
};
TEST_F(AccumulatorLayerTest, ShouldReset)
{
    acc.update_feature<true, WHITE>(0);
    acc.reset();
    ASSERT_EQ(acc.get_accumulator<WHITE>()[0], 2);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[1], -2);
}
TEST_F(AccumulatorLayerTest, ShouldActivateFeatureForWhite)
{
    ASSERT_EQ(acc.get_accumulator<WHITE>()[0], 2);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[1], -2);
    acc.update_feature<true, WHITE>(0);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[0], 7);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[1], -7);
}

TEST_F(AccumulatorLayerTest, ShouldDesactivateFeatureForWhite)
{
    ASSERT_EQ(acc.get_accumulator<WHITE>()[0], 2);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[1], -2);
    acc.update_feature<false, WHITE>(0);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[0], -3);
    ASSERT_EQ(acc.get_accumulator<WHITE>()[1], 3);
}

TEST_F(AccumulatorLayerTest, ShouldDesactivateFeatureForBlack)
{
    ASSERT_EQ(acc.get_accumulator<BLACK>()[0], 2);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[1], -2);
    acc.update_feature<false, BLACK>(0);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[0], -3);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[1], 3);
}
TEST_F(AccumulatorLayerTest, ShouldActivateFeatureForBlack)
{
    ASSERT_EQ(acc.get_accumulator<BLACK>()[0], 2);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[1], -2);
    acc.update_feature<true, BLACK>(0);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[0], 7);
    ASSERT_EQ(acc.get_accumulator<BLACK>()[1], -7);
}