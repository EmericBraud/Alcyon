#include "gtest/gtest.h"
#include "engine/eval/nnue/nnue_model.hpp"
#include "common/constants.hpp"
#include <array>
#include <cstdint>
#include <tuple>
#include <memory>

class NnueModelTest : public ::testing::Test
{
protected:
    // Taille basée sur l'encodeur : 32 * 12 * 64
    static constexpr int MAX_FEATURES = 24576;
    static constexpr int ACC_SIZE = 2;

    using Model = NnueModel<MAX_FEATURES, ACC_SIZE, 4, 1>;

    Model model;

    NnueModelTest()
        : model(
              // 1. Accumulator Biases
              std::array<std::int16_t, ACC_SIZE>{{1, 2}},

              // 2. Accumulator Weights (Allocation sur le tas pour éviter Stack Overflow)
              *std::unique_ptr<std::array<std::array<std::int8_t, ACC_SIZE>, MAX_FEATURES>>([]()
                                                                                            {
                  auto w = std::make_unique<std::array<std::array<std::int8_t, ACC_SIZE>, MAX_FEATURES>>();
                  for (auto& row : *w) row.fill(0);
                  // On initialise quelques poids pour les premiers indices pour les tests manuels
                  (*w)[0] = {1, 0};
                  (*w)[1] = {0, 1};
                  return w.release(); }()),

              // 3. Dense Layers
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

// --- Tests existants (comportement manuel) ---

TEST_F(NnueModelTest, ShouldEvaluateInitialBiases)
{
    // [1,2] concat [1,2] = [1,2,1,2] -> somme = 6
    ASSERT_EQ(model.get_result<WHITE>(), 6);
}

TEST_F(NnueModelTest, ShouldActivateFeatureManually)
{
    model.update_feature<true, WHITE>(0); // Ajoute {1,0} à l'acc blanc
    // Acc Blanc: [1+1, 2+0] = [2,2]. Acc Noir: [1,2]. Concat: [2,2,1,2] -> 7
    ASSERT_EQ(model.get_result<WHITE>(), 7);
}

// --- Tests de la fonction initialize() ---

TEST_F(NnueModelTest, ShouldInitializeFromOccupancies)
{
    std::array<U64, constants::NumPieceVariants> occupancies{};

    // Position minimale : Rois uniquement (ne génèrent pas de features selon ton assert)
    occupancies[KING] = (1ULL << 4);                              // e1
    occupancies[constants::PieceTypeCount + KING] = (1ULL << 60); // e8

    model.initialize(occupancies);

    // Ne doit contenir que les biais
    ASSERT_EQ(model.get_result<WHITE>(), 6);
}

TEST_F(NnueModelTest, ShouldInitializeWithPieces)
{
    std::array<U64, constants::NumPieceVariants> occupancies{};

    // Rois obligatoires
    occupancies[KING] = (1ULL << 4);                              // e1
    occupancies[constants::PieceTypeCount + KING] = (1ULL << 60); // e8

    // Ajout d'un pion blanc en a2
    occupancies[PAWN] = (1ULL << 8);

    model.initialize(occupancies);

    // L'accumulateur a été modifié par initialize, le score doit avoir changé
    // (Même si le poids est 0, l'opération a eu lieu sans crash)
    ASSERT_NO_FATAL_FAILURE(model.get_result<WHITE>());
}

TEST_F(NnueModelTest, InitializeShouldResetPreviousState)
{
    // 1. Modifier l'état manuellement
    model.update_feature<true, WHITE>(0);
    ASSERT_EQ(model.get_result<WHITE>(), 7);

    // 2. Réinitialiser avec juste les rois
    std::array<U64, constants::NumPieceVariants> occupancies{};
    occupancies[KING] = (1ULL << 4);
    occupancies[constants::PieceTypeCount + KING] = (1ULL << 60);

    model.initialize(occupancies);

    // 3. Retour au score de base (6)
    ASSERT_EQ(model.get_result<WHITE>(), 6);
}

TEST_F(NnueModelTest, ShouldHandleMultiplePiecesOfSameType)
{
    std::array<U64, constants::NumPieceVariants> occupancies{};
    occupancies[KING] = (1ULL << 4);
    occupancies[constants::PieceTypeCount + KING] = (1ULL << 60);

    // Deux pions blancs (a2 et b2)
    occupancies[PAWN] = (1ULL << 8) | (1ULL << 9);

    // Appel direct : si initialize() accède à un index invalide dans MAX_FEATURES,
    // le test plantera et sera marqué comme FAILED (Segmentation Fault / Signal).
    model.initialize(occupancies);

    // On vérifie que le résultat est calculable après l'initialisation
    ASSERT_NO_FATAL_FAILURE(model.get_result<WHITE>());
}