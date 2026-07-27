#include "engine/tt/transp_table.hpp"
#include "gtest/gtest.h"

class TTTest : public ::testing::Test
{
protected:
    // probe() a besoin d'un Board pour décompresser le move stocké ; pour un
    // move nul (Move()) la décompression n'y touche pas, un board par défaut
    // suffit.
    Board board;
    int tt_eval = TT_NO_EVAL;

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
    bool hit = tt.probe(board, key, 10, 0, -1000000, 1000000, retrieved_score, m, flag, tt_eval);

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
    tt.probe(board, key, 5, 0, -engine_constants::eval::Inf, engine_constants::eval::Inf, score, m, flag, tt_eval);
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
    bool hit = tt.probe(board, key, 10, 0, 60, 80, score, m, flag, tt_eval);
    ASSERT_TRUE(hit);
    ASSERT_LE(score, 60); // On vérifie que le score ne dépasse pas alpha
    ASSERT_EQ(score, 50); // En réalité, il doit retourner la valeur exacte stockée

    // Test 2 : Fenêtre [30, 40].
    // On sait que score <= 50. Est-ce que le score est <= 30 ? On ne sait pas.
    // Est-ce que le score est >= 40 ? On ne sait pas (il pourrait être 35).
    // On ne peut pas couper. hit doit être FALSE.
    hit = tt.probe(board, key, 10, 0, 30, 40, score, m, flag, tt_eval);
    ASSERT_FALSE(hit);
}
TEST_F(TTTest, StaticEvalRoundtrip)
{
    TranspositionTable tt;
    tt.resize(1);
    uint64_t key = 0x42;
    int score;
    Move m = 0;
    TTFlag flag;

    // Store avec éval statique : elle doit ressortir telle quelle, y compris
    // quand le probe ne produit pas de cutoff (fenêtre non résolvable).
    tt.store(key, 4, 0, 75, TT_ALPHA, Move(), -321);
    bool hit = tt.probe(board, key, 10, 0, 30, 40, score, m, flag, tt_eval);
    ASSERT_FALSE(hit); // depth stockée 4 < demandée 10
    ASSERT_EQ(tt_eval, -321);
}
TEST_F(TTTest, StoreWithoutEvalPreservesExistingEval)
{
    TranspositionTable tt;
    tt.resize(1);
    uint64_t key = 0x43;
    int score;
    Move m = 0;
    TTFlag flag;

    // 1. qsearch stocke une entrée avec éval.
    tt.store(key, 0, 0, 10, TT_EXACT, Move(), 512);
    // 2. negamax écrase la même clé, plus profond, sans éval : l'éval
    //    existante doit être héritée, pas effacée.
    tt.store(key, 8, 0, 99, TT_EXACT, Move());

    bool hit = tt.probe(board, key, 8, 0, -engine_constants::eval::Inf, engine_constants::eval::Inf, score, m, flag, tt_eval);
    ASSERT_TRUE(hit);
    ASSERT_EQ(score, 99);
    ASSERT_EQ(tt_eval, 512);

    // 3. Une entrée plus profonde déjà en place n'est pas remplacée par un
    //    store moins profond, mais l'éval fraîche doit y être greffée.
    uint64_t key2 = 0x44;
    tt.store(key2, 8, 0, 99, TT_EXACT, Move());
    tt.store(key2, 0, 0, 10, TT_EXACT, Move(), -77);
    hit = tt.probe(board, key2, 8, 0, -engine_constants::eval::Inf, engine_constants::eval::Inf, score, m, flag, tt_eval);
    ASSERT_TRUE(hit);
    ASSERT_EQ(score, 99); // l'entrée profonde a survécu
    ASSERT_EQ(tt_eval, -77);
}
