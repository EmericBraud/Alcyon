#include "gtest/gtest.h"

#ifdef SPSA_TUNING

#include <cmath>
#include <sstream>

#include "engine/config/config.hpp"
#include "interface/uci_option.hpp"

namespace
{
    struct SearchParamsSnapshot
    {
        int aspiration_enable_depth = engine_constants::search::aspiration::EnableDepth;
        int aspiration_mid_depth = engine_constants::search::aspiration::MidDepth;
        int aspiration_high_depth = engine_constants::search::aspiration::HighDepth;
        int aspiration_small_delta = engine_constants::search::aspiration::SmallDelta;
        int aspiration_mid_delta = engine_constants::search::aspiration::MidDelta;
        int aspiration_high_delta = engine_constants::search::aspiration::HighDelta;
        int aspiration_widen_min_delta = engine_constants::search::aspiration::WidenMinDelta;
        int aspiration_widen_max_delta = engine_constants::search::aspiration::WidenMaxDelta;
        int aspiration_max_iterations = engine_constants::search::aspiration::MaxIterations;
        int aspiration_mate_window_margin = engine_constants::search::aspiration::MateWindowMargin;

        int razoring_max_depth = engine_constants::search::razoring::MaxDepth;
        int nmp_r_const = engine_constants::search::null_move_pruning::RConst;
        int nmp_beta_margin = engine_constants::search::null_move_pruning::BetaMargin;
        double lmr_table_init_const = engine_constants::search::late_move_reduction::TableInitConst;

        ~SearchParamsSnapshot()
        {
            engine_constants::search::aspiration::EnableDepth = aspiration_enable_depth;
            engine_constants::search::aspiration::MidDepth = aspiration_mid_depth;
            engine_constants::search::aspiration::HighDepth = aspiration_high_depth;
            engine_constants::search::aspiration::SmallDelta = aspiration_small_delta;
            engine_constants::search::aspiration::MidDelta = aspiration_mid_delta;
            engine_constants::search::aspiration::HighDelta = aspiration_high_delta;
            engine_constants::search::aspiration::WidenMinDelta = aspiration_widen_min_delta;
            engine_constants::search::aspiration::WidenMaxDelta = aspiration_widen_max_delta;
            engine_constants::search::aspiration::MaxIterations = aspiration_max_iterations;
            engine_constants::search::aspiration::MateWindowMargin = aspiration_mate_window_margin;
            engine_constants::search::razoring::MaxDepth = razoring_max_depth;
            engine_constants::search::null_move_pruning::RConst = nmp_r_const;
            engine_constants::search::null_move_pruning::BetaMargin = nmp_beta_margin;
            engine_constants::search::late_move_reduction::TableInitConst = lmr_table_init_const;
        }
    };
} // namespace

TEST(SPSAUCIOptionTest, NmpBetaMarginUpdatesEngineParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::null_move_pruning::BetaMargin, "nmp_beta_margin");
    std::istringstream input("setoption name nmp_beta_margin value 240");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::null_move_pruning::BetaMargin, 240);
}

TEST(SPSAUCIOptionTest, AspirationOptionUpdatesEngineParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::aspiration::HighDelta, "aspiration_high_delta");
    std::istringstream input("setoption name aspiration_high_delta value 128");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::aspiration::HighDelta, 128);
}

TEST(SPSAUCIOptionTest, AspirationIterationsOptionUpdatesEngineParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::aspiration::MaxIterations, "aspiration_max_iterations");
    std::istringstream input("setoption name aspiration_max_iterations value 7");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::aspiration::MaxIterations, 7);
}

TEST(SPSAUCIOptionTest, IntOptionUpdatesEngineParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::null_move_pruning::RConst, "nmp_r_const");
    std::istringstream input("setoption name nmp_r_const value 6");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::null_move_pruning::RConst, 6);
}

TEST(SPSAUCIOptionTest, IntOptionRoundsFloatValue)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::null_move_pruning::RConst, "nmp_r_const");
    std::istringstream input("setoption name nmp_r_const value 6.6");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::null_move_pruning::RConst, 7);
}

TEST(SPSAUCIOptionTest, DoubleOptionUsesSPSAScale)
{
    SearchParamsSnapshot snapshot;

    UCIOption<double> option(&engine_constants::search::late_move_reduction::TableInitConst, "lmr_table_init_const");
    std::istringstream input("setoption name lmr_table_init_const value 175");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_DOUBLE_EQ(engine_constants::search::late_move_reduction::TableInitConst, 1.75);
}

TEST(SPSAUCIOptionTest, UnknownOptionDoesNotModifyParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::razoring::MaxDepth, "razoring_max_depth");
    std::istringstream input("setoption name wrong_option value 12");

    const bool parsed = option.parse_input(input);

    ASSERT_FALSE(parsed);
    ASSERT_EQ(engine_constants::search::razoring::MaxDepth, snapshot.razoring_max_depth);
}

TEST(SPSAUCIOptionTest, InvalidValueIsIgnored)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::razoring::MaxDepth, "razoring_max_depth");
    std::istringstream input("setoption name razoring_max_depth value invalid");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::razoring::MaxDepth, snapshot.razoring_max_depth);
}

#else

TEST(SPSAUCIOptionTest, DisabledWithoutSPSATuning)
{
    GTEST_SKIP() << "SPSA_TUNING not enabled";
}

#endif
