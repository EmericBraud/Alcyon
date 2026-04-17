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
        int razoring_max_depth = engine_constants::search::razoring::MaxDepth;
        int nmp_r_const = engine_constants::search::null_move_pruning::RConst;
        double lmr_table_init_const = engine_constants::search::late_move_reduction::TableInitConst;

        ~SearchParamsSnapshot()
        {
            engine_constants::search::razoring::MaxDepth = razoring_max_depth;
            engine_constants::search::null_move_pruning::RConst = nmp_r_const;
            engine_constants::search::late_move_reduction::TableInitConst = lmr_table_init_const;
        }
    };
} // namespace

TEST(SPSAUCIOptionTest, IntOptionUpdatesEngineParameter)
{
    SearchParamsSnapshot snapshot;

    UCIOption<int> option(&engine_constants::search::null_move_pruning::RConst, "nmp_r_const");
    std::istringstream input("setoption name nmp_r_const value 6");

    const bool parsed = option.parse_input(input);

    ASSERT_TRUE(parsed);
    ASSERT_EQ(engine_constants::search::null_move_pruning::RConst, 6);
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
