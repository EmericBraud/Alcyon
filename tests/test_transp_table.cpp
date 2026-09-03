#include "engine/tt/transp_table.hpp"
#include "gtest/gtest.h"

class TTTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};
TEST_F(TTTest, MateScoreConsistency)
{
    TranspositionTable tt;
    tt.resize(1);
    uint64_t key = 12345ULL;

    // Si on est au ply 5 et qu'on voit un mat dans 3 coups (score = MATE - 8)
    int score_found_at_ply_5 = engine_constants::eval::MateScore - 8;
    int ply_found = 5;

    tt.store(key, 10, ply_found, score_found_at_ply_5, TT_EXACT, Move());

    int retrieved_score;
    Move m = 0;
    TTFlag flag;
    // On sonde à la racine (ply 0)
    bool hit = tt.probe(key, 10, 0, -1000000, 1000000, retrieved_score, m, flag);

    ASSERT_TRUE(hit);

    // CORRECTION :
    // Le mat était à 3 coups de distance du ply 5 (8 - 5 = 3).
    // À la racine (ply 0), il est donc toujours à 3 coups de distance.
    // Le score attendu est engine::config::eval::MateScore - 3.
    ASSERT_EQ(retrieved_score, engine_constants::eval::MateScore - 3);
}
TEST_F(TTTest, DepthReplacement)
{
    TranspositionTable tt;
    tt.resize(1);
    uint64_t key = 0xABC;

    // 1. Stocke profondeur 5
    tt.store(key, 5, 0, 100, TT_EXACT, Move());

    // 2. Tente de stocker profondeur 3 sur la même clé
    tt.store(key, 3, 0, 200, TT_EXACT, Move());

    int score;
    Move m = 0;
    TTFlag flag;
    tt.probe(key, 5, 0, -engine_constants::eval::Inf, engine_constants::eval::Inf, score, m, flag);
    ASSERT_EQ(score, 100); // La profondeur 5 doit avoir été conservée car 5 > 3
}
TEST_F(TTTest, AlphaBetaCuts)
{
    TranspositionTable tt;
    tt.resize(1);
    uint64_t key = 0x1;
    int score;
    Move m = 0;
    TTFlag flag;

    // On stocke : "Le score est <= 50"
    tt.store(key, 10, 0, 50, TT_ALPHA, Move());

    // Test 1 : Fenêtre [60, 80].
    // Comme 50 <= 60, on sait que cette branche ne peut pas améliorer Alpha.
    // C'est un HIT, et le score retourné doit être <= Alpha.
    bool hit = tt.probe(key, 10, 0, 60, 80, score, m, flag);
    ASSERT_TRUE(hit);
    ASSERT_LE(score, 60); // On vérifie que le score ne dépasse pas alpha
    ASSERT_EQ(score, 50); // En réalité, il doit retourner la valeur exacte stockée

    // Test 2 : Fenêtre [30, 40].
    // On sait que score <= 50. Est-ce que le score est <= 30 ? On ne sait pas.
    // Est-ce que le score est >= 40 ? On ne sait pas (il pourrait être 35).
    // On ne peut pas couper. hit doit être FALSE.
    hit = tt.probe(key, 10, 0, 30, 40, score, m, flag);
    ASSERT_FALSE(hit);
}