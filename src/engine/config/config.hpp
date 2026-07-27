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
        // Sentinel for "no static eval recorded at this ply" (in-check
        // nodes) in SearchWorker::static_eval_stack -- see negamax.cpp's
        // improving heuristic. Far outside any real eval value (even NNUE's
        // internal scale, ~2.4x centipawns, stays well under 1000000).
        constexpr int NoStaticEval = -1000000;
    }
#ifdef NNUE_EVAL
    // Search parameters tuned for the NNUE evaluation
    namespace search
    {
        constexpr int MaxDepth = 64;

        namespace aspiration
        {
            PARAM_SPECIFIER int EnableDepth = 5;
            PARAM_SPECIFIER int MidDepth = 8;
            PARAM_SPECIFIER int HighDepth = 12;

            PARAM_SPECIFIER int SmallDelta = 16;
            PARAM_SPECIFIER int MidDelta = 41;
            PARAM_SPECIFIER int HighDelta = 20;

            PARAM_SPECIFIER int WidenMinDelta = 50;
            PARAM_SPECIFIER int WidenMaxDelta = 2000;
            PARAM_SPECIFIER int MaxIterations = 4;
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
            PARAM_SPECIFIER int MaxDepth = 4;
            PARAM_SPECIFIER int MarginDepthFactor = 73;
            PARAM_SPECIFIER int MarginConst = 61;
            // Subtracted from the margin when the static eval is improving
            // (this ply's vs ply-2's, same side to move): a position trending
            // up makes us trust a would-be beta cutoff more, so we accept a
            // smaller safety margin. See search::improving in negamax.cpp.
            PARAM_SPECIFIER int ImprovingMargin = 40;
        }
        namespace iterative_deepening
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int NewDepthIncr = 4;
        }
        namespace null_move_pruning
        {
            PARAM_SPECIFIER int MinDepth = 2;
            PARAM_SPECIFIER int RConst = 4;
            PARAM_SPECIFIER int RDiv = 4;
        }
        namespace futility_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 8;
            PARAM_SPECIFIER int MarginConst = 95;
            PARAM_SPECIFIER int MarginDepthFactor = 105;
            PARAM_SPECIFIER int ImprovingMargin = 40;
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
            PARAM_SPECIFIER int MaxDepthReduction = 1;

            PARAM_SPECIFIER double TableInitConst = 0.63065940599962;
            PARAM_SPECIFIER double TableInitDiv = 2.301959991800665;
        }
        namespace see_pruning
        {
            PARAM_SPECIFIER int MaxDepth = 6;
            PARAM_SPECIFIER int ThresholdDepthFactor = 15;
        }
    }
#else
    // Search parameters tuned for the HCE evaluation
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
            PARAM_SPECIFIER int ImprovingMargin = 40;
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
            PARAM_SPECIFIER int ImprovingMargin = 40;
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
#endif
}