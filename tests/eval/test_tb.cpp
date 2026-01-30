#include "gtest/gtest.h"
#include "engine/eval/tablebase.hpp"
#include "core/board/board.hpp"
#include "engine/eval/virtual_board.hpp"

class SyzygyTest : public ::testing::Test
{
protected:
    TableBase tb;
    VBoard b;

    void SetUp() override
    {
        if (static_cast<int>(TB_LARGEST) == 0)
        {
            GTEST_SKIP() << "Tablebases Syzygy non trouvées dans " << DATA_PATH "syzygy";
        }
    }
};

// Test 1: Position Gagnante (KRP vs KR - Lucena)
TEST_F(SyzygyTest, ProbeWDL_Win)
{
    b.load_fen("1r6/1P6/8/8/8/8/2R5/k1K5 w - - 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    auto result = tb.probe_wdl(b);
    ASSERT_EQ(result, TableBase::WDL_Result::WIN);
}

// Test 2: Position Perdante
TEST_F(SyzygyTest, ProbeWDL_Loss)
{
    b.load_fen("8/8/8/8/8/3k4/4q3/K7 w - - 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    auto result = tb.probe_wdl(b);
    ASSERT_EQ(result, TableBase::WDL_Result::LOSS);
}

// Test 3: Position Nulle (Finale Tour contre Tour)
TEST_F(SyzygyTest, ProbeWDL_Draw)
{
    b.load_fen("7k/8/8/8/8/8/1R6/K1r5 w - - 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    auto result = tb.probe_wdl(b);
    ASSERT_EQ(result, TableBase::WDL_Result::DRAW);
}

// Test 4: Root Probe (Trouver le coup gagnant)
TEST_F(SyzygyTest, ProbeRoot_FindsWinningMove)
{
    // Mat en 1 imminent : 6Q1/8/8/8/8/8/k7/2K5 w - - 0 1
    // Blanc (Dame en g3) va jouer pour mater ou gagner.
    b.load_fen("8/8/8/8/8/6Q1/8/k1K5 w - - 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    auto result = tb.probe_root(b);

    ASSERT_NE(result.move, 0);

    // CORRECTION ICI :
    // Votre SyzygyScore est 9000. Le score retourné sera 9000 - DTZ.
    // Donc on s'attend à > 8000, pas > 10000 (qui est le score de Mat Moteur).
    ASSERT_GT(result.score, 8000);
    ASSERT_LT(result.score, 10000); // Strictement inférieur au mat immédiat moteur

    ASSERT_TRUE(b.is_move_pseudo_legal(result.move));
    ASSERT_TRUE(b.is_move_legal(result.move));
}

// Test 5: Pat Immédiat (Stalemate)
TEST_F(SyzygyTest, ProbeWDL_ImmediateStalemate)
{
    // Blanc est Pat
    b.load_fen("8/8/8/8/8/8/5q2/7K w - - 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    // CORRECTION ICI :
    // Fathom renvoie FAIL sur une position terminale (Pat/Mat).
    // C'est au moteur de détecter le Pat via MoveGen avant d'appeler les TB.
    auto result = tb.probe_wdl(b);

    // On s'attend donc à FAIL (ou DRAW selon comment on veut interpréter,
    // mais techniquement Fathom dit "Je ne sais pas, c'est fini").
    // Si vous voulez que votre wrapper renvoie DRAW, il faut modifier TableBase::probe_wdl.
    // Pour l'instant, validons le comportement par défaut de Fathom :
    ASSERT_EQ(result, TableBase::WDL_Result::FAIL);
}

// Test 6: En-Passant Validity
TEST_F(SyzygyTest, ProbeWDL_EnPassant_NoCrash)
{
    b.load_fen("8/8/8/K7/3pP3/8/8/k7 b - e3 0 1");

    if (std::popcount(b.get_occupancy<NO_COLOR>()) > static_cast<int>(TB_LARGEST))
        GTEST_SKIP();

    auto result = tb.probe_wdl(b);
    ASSERT_NE(result, TableBase::WDL_Result::FAIL);
}