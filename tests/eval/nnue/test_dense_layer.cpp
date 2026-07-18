#include "gtest/gtest.h"
#include "engine/eval/nnue/dense_layer.hpp"
#include <array>
#include <cstdint>
class DenseLayerTest : public ::testing::Test
{
protected:
    // WeightScaleBits=0: hand-crafted small weights here aren't real quantized weights.
    DenseLayer<2, 1, 0> dense_layer;
    DenseLayerTest() : dense_layer(std::array<std::array<std::int8_t, 2>, 1>{{{{1, -1}}}}, std::array<std::int32_t, 1>{{1}})
    {
    }
};

TEST_F(DenseLayerTest, ShouldProcess)
{
    std::array<std::int8_t, 2> inputs = {5, 2};
    std::array<std::int8_t, 1> outputs = {68}; // We place a random value to make sure it gets erased
    dense_layer.process(inputs, outputs);
    ASSERT_EQ(outputs[0], 4);
}

TEST_F(DenseLayerTest, ShouldGetResult)
{
    std::array<std::int8_t, 2> inputs = {5, 2};
    ASSERT_EQ(dense_layer.get_result(inputs), 4);
}

TEST_F(DenseLayerTest, ShouldBeZeroWhenNegative)
{
    std::array<std::int8_t, 2> inputs = {0, 2};
    std::array<std::int8_t, 1> outputs = {68}; // We place a random value to make sure it gets erased
    dense_layer.process(inputs, outputs);
    ASSERT_EQ(outputs[0], 0);
}

TEST_F(DenseLayerTest, ShouldClampWhenTooLarge)
{
    std::array<std::int8_t, 2> inputs = {127, -127};
    std::array<std::int8_t, 1> outputs = {68}; // We place a random value to make sure it gets erased
    dense_layer.process(inputs, outputs);
    ASSERT_EQ(outputs[0], 127);
}