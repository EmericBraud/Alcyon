#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_model.hpp"
#include <array>
#include <cstdint>

// Minimal instantiation (Accum=4, 1 PSQT bucket, 1 layer-stack bucket, L2=2,
// L3=1) with hand-picked weights, so the v2 forward pass (pairwise-square L0,
// L1's last-two-columns skip term, L1+L2 squared-CReLU concatenated into a
// wider output layer, PSQT combination) can be verified by hand.
//
// Trace (no features activated, so both perspectives' accumulators equal the
// bias [200, 100, 150, 50]). Note compute_l0 pairs acc[i] with acc[i+Half]
// (not adjacent elements): out[0] uses acc[0] & acc[2], out[1] uses acc[1] &
// acc[3]. Per nnue-pytorch's double_feature_transform + ft_act quantization
// (ft_quantized_max=255, inference_l0_division_factor=512, and
// l0_correction_factor == 1 for this network's constants), each half is
// clamped to [0, 255] and the product is divided by 512, *not* by 128:
//   L0 pairwise square (Half=2): out[0] = clamp((200*150)//512, 0, 127) = 58
//                                 out[1] = clamp((100*50)//512, 0, 127) = 9
//   l0 = [58, 9, 58, 9]
//   L1 raw (weights [64,0,0,0] / [0,0,0,64]): raw[0] = 64*58 = 3712
//                                              raw[1] = 64*9 = 576
//   skip_raw = raw[0] - raw[1] = 3136
//   x0 = 3712>>7 = 29; sqr0 = (29*29)>>7 = 6
//   x1 = 576>>7 = 4; sqr1 = (4*4)>>7 = 0
//   l1_out = [6, 0, 29, 4]
//   L2 raw (weight all 64s, bias 0): raw = 64*(6+0+29+4) = 2496
//   x = 2496>>6 = 39; sqr = (39*39)>>7 = 11
//   l2_out = [11, 39]
//   l3_input = [6, 0, 29, 4, 11, 39]
//   output_raw (weights all 1s, bias 100) = 100 + 89 = 189
//   combined_raw = 189 + 3136 = 3325
//   layerstack_final = trunc(3325*75/128) = trunc(1948.2421875) = 1948
//   psqt_diff = 0
//   result = trunc((2*1948 + 0) / 32) = trunc(121.75) = 121
using TestModel = NnueModel</*Features=*/4, /*Accum=*/4, /*PsqtBuckets=*/1, /*LsBuckets=*/1, /*L2=*/2, /*L3=*/1>;

namespace
{
    TestModel make_model()
    {
        std::array<std::int16_t, 4> acc_biases{200, 100, 150, 50};
        std::array<std::array<std::int16_t, 4>, 4> acc_weights{};
        acc_weights[0] = {50, 0, 0, 0}; // feature 0 nudges the accumulator so update_feature is observable
        std::array<std::array<std::int32_t, 1>, 4> psqt_weights{};

        TestModel::L1Layer l1(
            std::array<std::array<std::int8_t, 4>, 2>{{{{64, 0, 0, 0}}, {{0, 0, 0, 64}}}},
            std::array<std::int32_t, 2>{{0, 0}});
        TestModel::L2Layer l2(
            std::array<std::array<std::int8_t, 4>, 1>{{{{64, 64, 64, 64}}}},
            std::array<std::int32_t, 1>{{0}});
        TestModel::OutputLayer output(
            std::array<std::array<std::int8_t, 6>, 1>{{{{1, 1, 1, 1, 1, 1}}}},
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

TEST(NnueModelTest, HandComputedForwardPassNoActiveFeatures)
{
    TestModel model = make_model();

    // No features activated: both perspectives see the same bias-only
    // accumulator, so both should produce the same result.
    EXPECT_EQ(model.get_result<WHITE>(/*piece_count=*/32), 121);
    EXPECT_EQ(model.get_result<BLACK>(/*piece_count=*/32), 121);
}

TEST(NnueModelTest, UpdateFeatureChangesResult)
{
    TestModel model = make_model();
    const auto baseline = model.get_result<WHITE>(32);

    model.update_feature<true, WHITE>(0);
    const auto after = model.get_result<WHITE>(32);

    EXPECT_NE(baseline, after);

    model.update_feature<false, WHITE>(0);
    EXPECT_EQ(model.get_result<WHITE>(32), baseline);
}
