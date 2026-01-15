#include "engine/eval/pos_eval.hpp"
#include "gtest/gtest.h"
#include "core/move/generator/move_generator.hpp"

class EvalStateTest : public ::testing::Test
{
protected:
    Board b;

    void SetUp() override
    {
    }
};

TEST_F(EvalStateTest, IncrementalConsistency)
{
    Board b;
    // Position complexe avec roques possibles, promotions et captures EP
    b.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    // On génère des coups au hasard ou via MoveGen
    MoveList list;
    MoveGen::generate_legal_moves(b, list);

    for (int i = 0; i < list.count; ++i)
    {
        Move m = list[i];

        // 1. Sauvegarde de l'état initial
        int initial_mg = b.get_eval_state().mg_pst[WHITE];
        uint64_t initial_pawn_key = b.get_eval_state().pawn_key;

        // 2. Jouer le coup
        b.play(m);

        // 3. Calcul "Brut" pour vérifier
        // On crée un EvalState tout neuf à partir du board actuel
        EvalState static_eval(b.get_all_bitboards());

        // 4. Vérification de l'incrément
        EXPECT_EQ(b.get_eval_state().mg_pst[WHITE], static_eval.mg_pst[WHITE])
            << "MG PST White incohérent après coup: " << m.to_uci();
        EXPECT_EQ(b.get_eval_state().mg_pst[BLACK], static_eval.mg_pst[BLACK]);
        EXPECT_EQ(b.get_eval_state().phase, static_eval.phase)
            << "Phase incohérente après coup: " << m.to_uci();
        EXPECT_EQ(b.get_eval_state().pawn_key, static_eval.pawn_key)
            << "Pawn Key incohérente après coup: " << m.to_uci();

        // 5. Unplay et vérification du retour à l'état initial
        b.unplay(m);
        EXPECT_EQ(b.get_eval_state().mg_pst[WHITE], initial_mg)
            << "Erreur de décrémentation PST après unplay: " << m.to_uci();
        EXPECT_EQ(b.get_eval_state().pawn_key, initial_pawn_key)
            << "Erreur de décrémentation PawnKey après unplay: " << m.to_uci();
    }
}

TEST_F(EvalStateTest, ConsistencyLongSequence)
{
    Board b;
    // Position de départ complexe
    b.load_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

    // Jouer 10 coups au hasard
    for (int i = 0; i < 10; i++)
    {
        MoveList list;
        MoveGen::generate_legal_moves(b, list);
        if (list.count == 0)
            break;

        Move m = list[0]; // On prend le premier coup pour le test
        b.play(m);
    }

    // On récupère l'état incrémental
    EvalState incremental = b.get_eval_state();

    // On force un recalcul total (Statique)
    EvalState static_eval(b.get_all_bitboards());

    // Vérifications
    ASSERT_EQ(incremental.mg_pst[WHITE], static_eval.mg_pst[WHITE]) << "Erreur MG PST White";
    ASSERT_EQ(incremental.eg_pst[WHITE], static_eval.eg_pst[WHITE]) << "Erreur EG PST White";
    ASSERT_EQ(incremental.phase, static_eval.phase) << "Erreur Phase";
    ASSERT_EQ(incremental.pawn_key, static_eval.pawn_key) << "Erreur Pawn Key";
}