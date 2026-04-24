#pragma once

#ifdef SPSA_TUNING
#define PARAM_SPECIFIER inline
#else
#define PARAM_SPECIFIER constexpr
#endif

namespace engine_constants
{
    namespace eval
    {
        constexpr int MateScore = 10000;
        constexpr int TacticalScore = 8500; // Move scoring
        constexpr int Inf = 10000;
        constexpr int SyzygyScore = 9000;
        constexpr int SyzygyMaxPieces = 5;
    }
    namespace search
    {
        constexpr int MaxDepth = 64;

        namespace aspiration
        {
            PARAM_SPECIFIER int EnableDepth = 5;
            PARAM_SPECIFIER int MidDepth = 8;
            PARAM_SPECIFIER int HighDepth = 12;

            PARAM_SPECIFIER int SmallDelta = 15;
            PARAM_SPECIFIER int MidDelta = 32;
            PARAM_SPECIFIER int HighDelta = 25;

            PARAM_SPECIFIER int WidenMinDelta = 50;
            PARAM_SPECIFIER int WidenMaxDelta = 2000;
            PARAM_SPECIFIER int MaxIterations = 5;
            PARAM_SPECIFIER int MateWindowMargin = 256;
        }

        namespace razoring
        {
            PARAM_SPECIFIER int MaxDepth = 3;
            PARAM_SPECIFIER int MarginDepthFactor = 100;
            PARAM_SPECIFIER int MarginConst = 0;
        }
        namespace reverse_futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 7;
            PARAM_SPECIFIER int MarginDepthFactor = 57;
            PARAM_SPECIFIER int MarginConst = 55;
        }
        namespace iterative_deepening
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int NewDepthIncr = 4;
        }
        namespace null_move_pruning
        {
            PARAM_SPECIFIER int MinDepth = 2;
            PARAM_SPECIFIER int RConst = 3;
            PARAM_SPECIFIER int RDiv = 5;
        }
        namespace futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 8;
            PARAM_SPECIFIER int MarginConst = 82;
            PARAM_SPECIFIER int MarginDepthFactor = 105;
        }
        namespace singular
        {
            PARAM_SPECIFIER int MinDepth = 8;
        }
        namespace null_move_reduction
        {
            PARAM_SPECIFIER int MaxDepth = 4;
            PARAM_SPECIFIER int MaxMovesConst = 8;
            PARAM_SPECIFIER int MaxMovesDepthSqFactor = 2;
        }
        namespace late_move_reduction
        {
            PARAM_SPECIFIER int MinDepth = 3;
            PARAM_SPECIFIER int MinMovesSearched = 5;
            PARAM_SPECIFIER int MaxDepthReduction = 2;

            PARAM_SPECIFIER double TableInitConst = 0.6295;
            PARAM_SPECIFIER double TableInitDiv = 2.3783;
        }
        namespace see_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int ThresholdDepthFactor = 15;
        }
    }
}