
#include "gtest/gtest.h"
#include "engine/eval/virtual_board.hpp"
#include "engine/eval/nnue/features_encoder.hpp"
#include <iostream>

#ifdef NNUE_EVAL

TEST(FeatureIndexDebug, PrintQueenIndices)
{
    VBoard a; // with queen
    VBoard b; // without queen
    ASSERT_TRUE(a.load_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K2Q w - - 0 1"));
    ASSERT_TRUE(b.load_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1"));

    auto occs_a = a.get_all_bitboards();
    auto occs_b = b.get_all_bitboards();

    int white_king_a = cpu::pop_lsb(occs_a[KING]);
    int white_king_b = cpu::pop_lsb(occs_b[KING]);

    // find queen square in a
    int queen_sq = cpu::pop_lsb(occs_a[QUEEN]);

    int idx_a_w = feature_encoder::get_feature_index<WHITE>(white_king_a, BLACK, QUEEN, queen_sq);
    int idx_a_b = feature_encoder::get_feature_index<BLACK>(cpu::pop_lsb(occs_a[constants::PieceTypeCount + KING]), BLACK, QUEEN, queen_sq);

    std::cout << "white king a=" << white_king_a << " queen_sq=" << queen_sq << " idx_a_w=" << idx_a_w << " idx_a_b=" << idx_a_b << std::endl;

    // For b there is no queen, but compute an example index for square a1 (0)
    int idx_b_w = feature_encoder::get_feature_index<WHITE>(white_king_b, BLACK, QUEEN, 0);
    std::cout << "white king b=" << white_king_b << " idx_b_w(example a1)=" << idx_b_w << std::endl;

    // Sanity: ensure indices are within expected range
    EXPECT_GE(idx_a_w, 0);
    EXPECT_GE(idx_a_b, 0);
}

#else

TEST(FeatureIndexDebug, RequiresNnueBuild)
{
    GTEST_SKIP() << "NNUE_EVAL not enabled in this build.";
}

#endif
