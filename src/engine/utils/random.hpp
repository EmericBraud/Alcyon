#pragma once

#include <cstdint>

namespace engine::random
{
    static inline uint64_t splitmix64(uint64_t x) // Used for deterministic random noise generation in negamax move ordering
    {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }
}
