#pragma once

namespace engine::config
{
    namespace eval
    {
        constexpr int MateScore = 10000;
        constexpr int TacticalScore = 8500;
        constexpr int Inf = 10000;
    }
    namespace search
    {
        constexpr int MaxDepth = 64;
    }
}