#pragma once

#ifdef __SSE2__
#include <xmmintrin.h>
#endif

#include "core/utils/mask.hpp"
namespace core::cpu
{
    template <typename T>
    inline void prefetch(const T *ptr, bool write = false, int locality = 3)
    {
        __builtin_prefetch(static_cast<const void *>(ptr), write ? 1 : 0, locality);
    }

    inline int get_lsb_index(U64 bb)
    {
        return __builtin_ctzll(bb);
    }

    inline int pop_lsb(U64 &bb)
    {
        int s = get_lsb_index(bb);
        bb &= bb - 1;
        return s;
    }
}