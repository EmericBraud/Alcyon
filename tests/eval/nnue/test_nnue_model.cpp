#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_model.hpp"
#include "common/constants.hpp"
#include <array>
#include <cstdint>
#include <tuple>

// Minimal instantiation (Accum=4, 1 PSQT bucket, 1 layer-stack bucket, L2=1,
// L3=1) with hand-picked weights, so the full forward pass (pairwise-square
// L0, layer-stack skip connection, PSQT combination) can be verified by hand.
//
// Trace (no features activated, so both perspectives' accumulators equal the
// bias [10, 20, 5, 8]):
//   clamp -> [10, 20, 5, 8]
//   L0 pairwise square (Half=2): out[0] = (10*5*127)>>7 = 49
//                                 out[1] = clamp((20*8*127)>>7, 0, 127) = clamp(158,0,127) = 127
//   l0 = [49, 127, 49, 127]                        (us-half, them-half; identical here)
//   L1 raw: neuron0 (weight [64,0,0,0]) = 64*49 = 3136
//           neuron1/skip (weight [0,0,0,64]) = 64*127 = 8128
//   x = 3136>>6 = 49; sqr = (49*49)>>7 = 18; l1_out = [18, 49]
//   L2 (weight [64,64]): acc = 64*(18+49) = 4288; >>6 = 67
//   output (weight [2], bias 100): 100 + 2*67 = 234
//   skip_rescaled = 8128*150/127 = 9600
//   layerstack_final = 234 + 9600 = 9834
//   psqt_diff = 0 (no features activated)
//   result = (2*9834 + 0) / 32 = 19668/32 = 614
using TestModel = NnueModel</*Features=*/4, /*Accum=*/4, /*PsqtBuckets=*/1, /*LsBuckets=*/1, /*L2=*/1, /*L3=*/1>;

namespace
{
    TestModel make_model()
    {
        std::array<std::int16_t, 4> acc_biases{10, 20, 5, 8};
        std::array<std::array<std::int16_t, 4>, 4> acc_weights{};
        std::array<std::array<std::int32_t, 1>, 4> psqt_weights{};

        TestModel::L1Layer l1(
            std::array<std::array<std::int8_t, 4>, 2>{{{{64, 0, 0, 0}}, {{0, 0, 0, 64}}}},
            std::array<std::int32_t, 2>{{0, 0}});
        TestModel::L2Layer l2(
            std::array<std::array<std::int8_t, 2>, 1>{{{{64, 64}}}},
            std::array<std::int32_t, 1>{{0}});
        TestModel::OutputLayer output(
            std::array<std::array<std::int8_t, 1>, 1>{{{{2}}}},
            std::array<std::int32_t, 1>{{100}});

        std::array<TestModel::LayerStackBucket, 1> layer_stacks{
            TestModel::LayerStackBucket(std::move(l1), std::move(l2), std::move(output))};

        return TestModel(
            std::move(acc_biases),
            std::move(acc_weights),
            std::move(psqt_weights),
            std::move(layer_stacks));
    }
}

TEST(NnueModelTest, HandComputedForwardPass)
{
    TestModel model = make_model();

    ASSERT_EQ(model.get_result<WHITE>(/*piece_count=*/2), 614);
    ASSERT_EQ(model.get_result<BLACK>(/*piece_count=*/2), 614);
}

TEST(NnueModelTest, FeatureActivationIsReversible)
{
    std::array<std::int16_t, 4> acc_biases{10, 20, 5, 8};
    std::array<std::array<std::int16_t, 4>, 4> acc_weights{};
    acc_weights[0] = {1, 2, 3, 4};
    std::array<std::array<std::int32_t, 1>, 4> psqt_weights{};
    psqt_weights[0] = {7};

    TestModel::L1Layer l1(
        std::array<std::array<std::int8_t, 4>, 2>{{{{64, 0, 0, 0}}, {{0, 0, 0, 64}}}},
        std::array<std::int32_t, 2>{{0, 0}});
    TestModel::L2Layer l2(
        std::array<std::array<std::int8_t, 2>, 1>{{{{64, 64}}}},
        std::array<std::int32_t, 1>{{0}});
    TestModel::OutputLayer output(
        std::array<std::array<std::int8_t, 1>, 1>{{{{2}}}},
        std::array<std::int32_t, 1>{{100}});
    std::array<TestModel::LayerStackBucket, 1> layer_stacks{
        TestModel::LayerStackBucket(std::move(l1), std::move(l2), std::move(output))};

    TestModel model(
        std::move(acc_biases),
        std::move(acc_weights),
        std::move(psqt_weights),
        std::move(layer_stacks));

    const auto initial = model.get_result<WHITE>(2);

    model.update_feature<true, WHITE>(0);
    ASSERT_EQ(model.get_accumulator().get_accumulator<WHITE>()[0], 11);
    ASSERT_NE(model.get_result<WHITE>(2), initial);

    model.update_feature<false, WHITE>(0);
    ASSERT_EQ(model.get_accumulator().get_accumulator<WHITE>()[0], 10);
    ASSERT_EQ(model.get_result<WHITE>(2), initial);
}
