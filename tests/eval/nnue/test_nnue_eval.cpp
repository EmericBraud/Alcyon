#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_eval.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include <array>
#include <tuple>
#include <memory>

class NnueEvalTest : public ::testing::Test
{
    // On définit une version de test avec des petites dimensions
    using TestEval = NnueEval<2, 2, 4, 4, 1>;
    using Model = TestEval::Model;

    void SetUp() override
    {
        auto dense_layers = std::make_tuple(
            DenseLayer<4, 4>(
                std::array<std::array<std::int8_t, 4>, 4>{{{{1, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{0, 0, 0, 1}}}},
                std::array<std::int32_t, 4>{{0, 0, 0, 0}}),

            DenseLayer<4, 4>(
                std::array<std::array<std::int8_t, 4>, 4>{{{{1, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 1, 0}}, {{0, 0, 0, 1}}}},
                std::array<std::int32_t, 4>{{0, 0, 0, 0}}),

            DenseLayer<4, 1>(
                std::array<std::array<std::int8_t, 4>, 1>{{{{1, 1, 1, 1}}}},
                std::array<std::int32_t, 1>{{0}}));

        Model model = Model(
            std::array<std::int16_t, 2>{{1, 2}},
            std::array<std::array<std::int8_t, 2>, 2>{{{{1, 0}}, {{0, 1}}}},
            std::move(dense_layers));

        eval_to_test = std::make_unique<TestEval>(std::move(model));
    }

protected:
    std::unique_ptr<TestEval> eval_to_test;
};
TEST_F(NnueEvalTest, ShouldDisableFeature)
{
    ASSERT_EQ(eval_to_test.get()->get_accumulator().get_accumulator<WHITE>()[0], 1);
    eval_to_test.get()->update_feature<false, WHITE>(0);
    ASSERT_EQ(eval_to_test.get()->get_accumulator().get_accumulator<WHITE>()[0], 0);
}
TEST_F(NnueEvalTest, ShouldActivateFeature)
{
    // Valeur initiale (biais)
    ASSERT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 1);

    // Activation de la feature 0 (Poids = 1 pour l'index 0)
    eval_to_test->update_feature<true, WHITE>(0);

    // 1 (biais) + 1 (poids) = 2
    EXPECT_EQ(eval_to_test->get_accumulator().get_accumulator<WHITE>()[0], 2);
}
TEST_F(NnueEvalTest, ShouldCalculateCorrectFinalScore)
{
    // Reset complet (on part des biais)
    // Accu White = {1, 2}, Accu Black = {1, 2} (si non modifié)
    // Comme ton evaluate_abs concatène probablement les deux (2+2=4 entrées)
    // L1 & L2 = Identité, L3 = Somme des 4 entrées

    // Score attendu : 1 + 2 + 1 + 2 = 6
    int32_t score = eval_to_test->evaluate_abs();
    EXPECT_EQ(score, 6);

    // Si on active la feature 1 (Poids {0, 1})
    eval_to_test->update_feature<true, WHITE>(1);
    // L'accu White devient {1, 2+1=3}. Score : 1 + 3 + 1 + 2 = 7
    EXPECT_EQ(eval_to_test->evaluate_abs(), 7);
}
TEST_F(NnueEvalTest, IncrementalUpdateShouldBeReversible)
{
    int32_t initial_score = eval_to_test->evaluate_abs();

    // 1. On modifie l'état
    eval_to_test->update_feature<true, WHITE>(0);
    ASSERT_NE(initial_score, eval_to_test->evaluate_abs()); // On s'assure que ça a bougé

    // 2. On fait l'opération inverse exacte
    eval_to_test->update_feature<false, WHITE>(0);

    // 3. On DOIT revenir à l'original
    EXPECT_EQ(initial_score, eval_to_test->evaluate_abs())
        << "L'accumulateur n'est pas revenu a son etat initial apres un add/remove !";
}
TEST_F(NnueEvalTest, ShouldChangeScoreWhenFeatureIsAdded)
{
    int32_t initial_score = eval_to_test->evaluate_abs();

    // On active la feature 0 (Poids {1, 0})
    eval_to_test->update_feature<true, WHITE>(0);

    // Le score doit être différent (6 -> 7)
    EXPECT_NE(initial_score, eval_to_test->evaluate_abs());
    EXPECT_EQ(eval_to_test->evaluate_abs(), 7);
}