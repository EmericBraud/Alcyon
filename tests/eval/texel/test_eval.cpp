#ifdef TEXEL_TUNING

#include "engine/eval/pos_eval.hpp"
#include "engine/eval/tuning/eval_features.hpp"
#include "engine/eval/virtual_board.hpp"
#include "engine/config/eval.hpp"
#include "gtest/gtest.h"
#include <string>
#include <array>

class TexelTuningEvalTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(TexelTuningEvalTest, EvalConsistency)
{
    const std::array<std::string, 2> fens = {"8/8/1kq5/8/4R3/3QK3/8/8 w - - 0 1", "rn1qkbnr/pp2pppp/2p5/5b2/3PN3/8/PPP2PPP/R1BQKBNR w KQkq - 0 1"};
    for (auto fen : fens)
    {
        VBoard b;
        b.load_fen(fen);

        int eval_score = Eval::eval(b, -engine_constants::eval::Inf, engine_constants::eval::Inf);
        EvalFeatures f = Eval::extract_eval_features(b);
        int texel_eval = static_cast<int>(Eval::score_eval_features(f, b));
        ASSERT_EQ(texel_eval, eval_score);
    }
}

#endif