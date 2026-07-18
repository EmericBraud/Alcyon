#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_eval.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include <array>
#include <tuple>
#include <memory>
#include <fstream>
#include <cstdio>

// Same tiny hand-computed dimensions as test_nnue_model.cpp: Accum=4, 1 PSQT
// bucket, 1 layer-stack bucket, L2=1, L3=1. See that file for the full trace.
using TestEval = NnueEval</*Features=*/4, /*Accum=*/4, /*PsqtBuckets=*/1, /*LsBuckets=*/1, /*L2=*/1, /*L3=*/1>;
using TestModel = TestEval::Model;

namespace
{
    TestModel make_model(std::array<std::array<std::int16_t, 4>, 4> acc_weights = {})
    {
        std::array<std::int16_t, 4> acc_biases{10, 20, 5, 8};
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

class NnueEvalTest : public ::testing::Test
{
protected:
    std::unique_ptr<TestEval> eval_to_test;

    void SetUp() override
    {
        std::array<std::array<std::int16_t, 4>, 4> acc_weights{};
        acc_weights[0] = {1, 0, 0, 0};
        eval_to_test = std::make_unique<TestEval>(make_model(acc_weights));
    }
};

TEST_F(NnueEvalTest, ShouldDisableFeature)
{
    eval_to_test->update_feature<true, WHITE>(0);
    ASSERT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 11);
    eval_to_test->update_feature<false, WHITE>(0);
    ASSERT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 10);
}

TEST_F(NnueEvalTest, ShouldActivateFeature)
{
    ASSERT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 10);
    eval_to_test->update_feature<true, WHITE>(0);
    ASSERT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 11);
}

TEST_F(NnueEvalTest, ShouldCalculateCorrectFinalScore)
{
    // No feature activated yet: matches the hand-computed trace in
    // test_nnue_model.cpp (bias-only accumulators -> score 614).
    EXPECT_EQ(eval_to_test->evaluate_abs(WHITE, /*piece_count=*/2), 614);

    eval_to_test->update_feature<true, WHITE>(0);
    EXPECT_NE(eval_to_test->evaluate_abs(WHITE, 2), 614);
}

TEST_F(NnueEvalTest, IncrementalUpdateShouldBeReversible)
{
    const int32_t initial_score = eval_to_test->evaluate_abs(WHITE, 2);

    eval_to_test->update_feature<true, WHITE>(0);
    ASSERT_NE(initial_score, eval_to_test->evaluate_abs(WHITE, 2));

    eval_to_test->update_feature<false, WHITE>(0);
    EXPECT_EQ(initial_score, eval_to_test->evaluate_abs(WHITE, 2))
        << "L'accumulateur n'est pas revenu a son etat initial apres un add/remove !";
}

TEST_F(NnueEvalTest, ShouldLoadRealModelFile)
{
    // On utilise les vraies dimensions du modele entraine (32 king buckets *
    // 11 piece types * 64 squares), et l'architecture reelle (squared-CReLU +
    // skip connection + PSQT + 8 layer-stack buckets).
    using RealEval = NnueEval<22528, 256, 8, 8, 32, 32>;

    const std::string path = "nnue/v1.nnue";

    EXPECT_NO_FATAL_FAILURE({
        RealEval eval(path);

        // Un score ne doit jamais etre exactement 0 sur une position initialisee (biais).
        EXPECT_NE(eval.evaluate_abs(WHITE, /*piece_count=*/32), 0);
    });
}

TEST_F(NnueEvalTest, ShouldHandleMissingFile)
{
    ASSERT_DEATH({ TestEval eval("non_existent_file.nnue"); }, ".*Could not open NNUE file.*");
}

TEST_F(NnueEvalTest, ShouldDetectTruncatedFile)
{
    const std::string truncated_path = "truncated_test.nnue";

    {
        std::ofstream out(truncated_path, std::ios::binary);
        uint32_t dummy = 1234;
        out.write(reinterpret_cast<char *>(&dummy), sizeof(dummy));
        out.close();
    }

    ASSERT_DEATH({ TestEval eval(truncated_path); }, ".*NNUE read failed.*");

    std::remove(truncated_path.c_str());
}
