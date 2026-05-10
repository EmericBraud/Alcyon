#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_eval.hpp"
#include "engine/eval/nnue/nnue_model.hpp"
#include <array>
#include <tuple>
#include <memory>
#include <fstream>
#include <cstdio>

class NnueEvalTest : public ::testing::Test
{
protected:
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
TEST_F(NnueEvalTest, ShouldLoadRealModelFile)
{
    // On utilise les vraies dimensions du modèle entraîné
    using RealEval = NnueEval<24576, 256, 32, 32, 1>;

    const std::string path = "nnue/v1.nnue";

    // Test de chargement
    // Si le fichier n'existe pas dans le path spécifié, le test échouera via FATAL
    EXPECT_NO_FATAL_FAILURE({
        RealEval eval(path);

        // Un score ne doit jamais être exactement 0 sur une position initialisée (biais)
        // Cela permet de vérifier que l'on n'a pas chargé un modèle vide
        EXPECT_NE(eval.evaluate_abs(), 0);
    });
}

TEST_F(NnueEvalTest, ShouldHandleMissingFile)
{
    // On s'attend à ce que le moteur appelle FATAL (ASSERT_DEATH capture l'arrêt du programme)
    ASSERT_DEATH({ TestEval eval("non_existent_file.nnue"); }, ".*Could not open NNUE file.*");
}

TEST_F(NnueEvalTest, ShouldDetectTruncatedFile)
{
    const std::string truncated_path = "truncated_test.nnue";

    // Création d'un fichier binaire corrompu (trop court)
    {
        std::ofstream out(truncated_path, std::ios::binary);
        uint32_t dummy = 1234;
        out.write(reinterpret_cast<char *>(&dummy), sizeof(dummy));
        out.close();
    }

    // Le test vérifie que le chargement échoue proprement car le fichier est trop court
    // pour remplir les structures de données
    ASSERT_DEATH({ TestEval eval(truncated_path); }, ".*Invalid or truncated NNUE file.*");

    // Nettoyage
    std::remove(truncated_path.c_str());
}

TEST_F(NnueEvalTest, HeaderOffsetVerification)
{
    const std::string header_test_path = "header_test.nnue";

    // On crée un fichier qui contient 128 octets de junk, puis des données connues
    {
        std::ofstream out(header_test_path, std::ios::binary);
        char junk[128] = {0};
        out.write(junk, 128);

        // Simulation de biais (Accum = 2)
        int16_t biases[2] = {42, 84};
        out.write(reinterpret_cast<char *>(biases), sizeof(biases));

        // Remplissage minimal pour passer les lectures suivantes
        // Poids (Features 2 * Accum 2)
        int8_t weights[4] = {0};
        out.write(reinterpret_cast<char *>(weights), sizeof(weights));

        // Dense Layers (le reste du fichier)
        // On remplit avec suffisamment de zéros pour que read_dense_layer ne fail pas
        char padding[1000] = {0};
        out.write(padding, 1000);

        out.close();
    }

    // Si l'offset de 128 est correct, l'accumulateur doit avoir chargé {42, 84}
    TestEval eval(header_test_path);
    auto acc = eval.get_accumulator().get_accumulator<WHITE>();

    EXPECT_EQ(acc[0], 42);
    EXPECT_EQ(acc[1], 84);

    std::remove(header_test_path.c_str());
}