#pragma once

#ifdef __SSE2__
#include <xmmintrin.h>
#endif

#if defined(_MSC_VER)
#define FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FORCE_INLINE inline __attribute__((always_inline))
#else
#define FORCE_INLINE inline
#endif

#include "common/mask.hpp"
namespace cpu
{
    template <typename T, bool Write = false, int Locality = 3>
    FORCE_INLINE void prefetch(const T *ptr)
    {
        static_assert(Locality >= 0 && Locality <= 3, "locality must be 0..3");
        __builtin_prefetch(static_cast<const void *>(ptr), Write ? 1 : 0, Locality);
    }

    FORCE_INLINE int get_lsb_index(U64 bb)
    {
        return __builtin_ctzll(bb);
    }

    FORCE_INLINE int pop_lsb(U64 &bb)
    {
        int s = get_lsb_index(bb);
        bb &= bb - 1;
        return s;
    }
}