#pragma once

namespace engine::config
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
    }
}