#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_model.hpp"

#include <array>
#include <cstdint>
#include <tuple>

class NnueModelTest : public ::testing::Test
{
protected:
    using Model = NnueModel<2, 2, 4, 1>;

    Model model;

    NnueModelTest()
        : model(
              std::array<std::int16_t, 2>{{1, 2}}, // accumulator biases

              std::array<std::array<std::int8_t, 2>, 2>{{{{1, 0}},
                                                         {{0, 1}}}}, // accumulator weights

              std::make_tuple(
                  DenseLayer<4, 4>(
                      std::array<std::array<std::int8_t, 4>, 4>{{{{1, 0, 0, 0}},
                                                                 {{0, 1, 0, 0}},
                                                                 {{0, 0, 1, 0}},
                                                                 {{0, 0, 0, 1}}}},
                      std::array<std::int32_t, 4>{{0, 0, 0, 0}}),

                  DenseLayer<4, 1>(
                      std::array<std::array<std::int8_t, 4>, 1>{{{{1, 1, 1, 1}}}},
                      std::array<std::int32_t, 1>{{0}})))
    {
    }
};

TEST_F(NnueModelTest, ShouldEvaluateInitialBiasesForWhite)
{
    ASSERT_EQ(model.get_result<WHITE>(), 6);
}

TEST_F(NnueModelTest, ShouldEvaluateInitialBiasesForBlack)
{
    ASSERT_EQ(model.get_result<BLACK>(), 6);
}

TEST_F(NnueModelTest, ShouldActivateFeatureForWhitePerspective)
{
    model.update_feature<true, WHITE>(0);

    // WHITE acc: [1,2] + [1,0] = [2,2]
    // BLACK acc: [1,2]
    // concat: [2,2,1,2]
    // sum = 7
    ASSERT_EQ(model.get_result<WHITE>(), 7);
}

TEST_F(NnueModelTest, ShouldActivateFeatureForBlackPerspective)
{
    model.update_feature<true, BLACK>(1);

    // WHITE acc: [1,2]
    // BLACK acc: [1,2] + [0,1] = [1,3]
    // concat from WHITE: [white, black] = [1,2,1,3]
    ASSERT_EQ(model.get_result<WHITE>(), 7);
}

TEST_F(NnueModelTest, ShouldUsePerspectiveOrder)
{
    model.update_feature<true, WHITE>(0);

    // get_result<WHITE>: [white, black] = [2,2,1,2]
    ASSERT_EQ(model.get_result<WHITE>(), 7);

    // get_result<BLACK>: [black, white] = [1,2,2,2]
    // Avec une couche identité + somme finale, le résultat reste 7.
    ASSERT_EQ(model.get_result<BLACK>(), 7);
}

TEST_F(NnueModelTest, ShouldDisableFeature)
{
    model.update_feature<true, WHITE>(0);
    ASSERT_EQ(model.get_result<WHITE>(), 7);

    model.update_feature<false, WHITE>(0);
    ASSERT_EQ(model.get_result<WHITE>(), 6);
}

TEST_F(NnueModelTest, ShouldClampAccumulatorBeforeDenseLayers)
{
    model.update_feature<true, WHITE>(0);
    model.update_feature<true, WHITE>(0);

    // WHITE acc: [1,2] + 2*[1,0] = [3,2]
    // BLACK acc: [1,2]
    // concat: [3,2,1,2]
    ASSERT_EQ(model.get_result<WHITE>(), 8);
}