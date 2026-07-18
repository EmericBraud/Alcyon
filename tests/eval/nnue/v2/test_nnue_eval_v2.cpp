#include "gtest/gtest.h"
#include "engine/eval/nnue/v2/nnue_eval_v2.hpp"
#include "core/board/board.hpp"
#include "common/constants.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>

// End-to-end smoke test against the real data/nnue/v2.nnue file: loads it
// (exercising the segmented Full_Threats(int8)/HalfKAv2_hm^(int16) weight
// tensor parsing and LEB128 decompression), builds both perspectives'
// accumulators for the starting position via a full board scan, and checks
// the result is finite and non-trivial. This does *not* validate the network
// produces a *correct* evaluation (no reference/oracle is available without
// the Python trainer + torch), only that the whole pipeline runs without
// crashing and produces plausible output.
TEST(NnueEvalV2Test, LoadsRealModelAndEvaluatesStartingPosition)
{
    const std::string path = "nnue/v2.nnue";

    std::ifstream probe(path, std::ios::binary);
    if (!probe)
        GTEST_SKIP() << "data/nnue/v2.nnue not found at '" << path << "' (untracked binary, expected in dev checkouts)";
    probe.close();

    nnue_v2::NnueEvalV2 eval(path);

    Board board;
    ASSERT_TRUE(board.load_fen(constants::FenInitPos));

    EXPECT_NO_FATAL_FAILURE({
        eval.initialize(board);
        const std::int32_t score = eval.evaluate_abs(WHITE, /*piece_count=*/32);
        // Sanity bound: a legitimate NNUE eval of the starting position should
        // be a small number of centipawns, not something wildly out of range
        // (which would indicate a scale/format bug).
        EXPECT_LT(std::abs(score), 2000);
    });
}

TEST(NnueEvalV2Test, ShouldHandleMissingFile)
{
    ASSERT_DEATH({ nnue_v2::NnueEvalV2 eval("non_existent_v2_file.nnue"); }, ".*Could not open NNUE v2 file.*");
}
