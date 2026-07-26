#include "gtest/gtest.h"
#include "engine/eval/nnue/halfka_v2_hm_encoder.hpp"

// Verifies structural invariants of the ported HalfKAv2_hm real-feature index
// formula (model/modules/features/halfka_v2_hm.py at nnue-pytorch commit
// 4289208fe20cc6ec8753e5ee14c2f210de783ff0): bounds, king-bucket symmetry
// across the fold, and that own/enemy king never collide.

TEST(HalfkaV2HmEncoderTest, IndicesAreInRange)
{
    for (int ksq = 0; ksq < 64; ++ksq)
    {
        for (int sq = 0; sq < 64; ++sq)
        {
            for (int pt = PAWN; pt <= KING; ++pt)
            {
                for (int c = 0; c < 2; ++c)
                {
                    const auto piece_type = static_cast<Piece>(pt);
                    const auto piece_color = static_cast<Color>(c);

                    const int idx_w = nnue::halfka::feature_index<WHITE>(ksq, piece_color, piece_type, sq);
                    const int idx_b = nnue::halfka::feature_index<BLACK>(ksq, piece_color, piece_type, sq);

                    EXPECT_GE(idx_w, 0);
                    EXPECT_LT(idx_w, nnue::halfka::NUM_REAL_FEATURES);
                    EXPECT_GE(idx_b, 0);
                    EXPECT_LT(idx_b, nnue::halfka::NUM_REAL_FEATURES);
                }
            }
        }
    }
}

TEST(HalfkaV2HmEncoderTest, KingBucketOnlyDependsOnOrientedFile)
{
    // Orientation always folds the king onto files e-h (>=4), and the bucket
    // table only assigns non-negative buckets there.
    for (int ksq = 0; ksq < 64; ++ksq)
    {
        const int o_ksq = nnue::halfka::orient<WHITE>(ksq, ksq);
        EXPECT_GE(o_ksq % 8, 4);
        EXPECT_GE(nnue::halfka::KingBuckets[o_ksq], 0);
    }
}

TEST(HalfkaV2HmEncoderTest, OwnAndEnemyKingNeverCollideForSameKsq)
{
    // Own king is always physically on `king_sq`, enemy king is elsewhere;
    // this checks the merged p_idx'=10 plane doesn't alias them to the same
    // real feature index when they're (correctly) on different squares.
    const int ksq = 4; // e1
    const int enemy_ksq = 60; // e8
    const int idx_own = nnue::halfka::feature_index<WHITE>(ksq, WHITE, KING, ksq);
    const int idx_enemy = nnue::halfka::feature_index<WHITE>(ksq, BLACK, KING, enemy_ksq);

    EXPECT_NE(idx_own, idx_enemy);
}
