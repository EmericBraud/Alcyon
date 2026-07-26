// dot_product.hpp
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#endif

// dot(u8[N], i8[N]) -> i32 : the single hot primitive behind every dense
// layer of the network (see dense_layer.hpp).
//
// CONTRACT: a[i] must be in [0, 127] (activations come out of a clamp --
// ClippedReLU / squared-CReLU / pairwise-square all saturate at 127) and
// b[i] in [-128, 127]. The [0, 127] bound is what makes every fast path
// below exact:
//   - x86 vpdpbusd is unsigned x signed, so `a` has to be non-negative;
//   - x86 vpmaddubsw saturates its i16 pair-sums, and 127*128 + 127*128 =
//     32512 < 32767 means no pair can ever saturate;
//   - NEON sdot is signed x signed, and [0, 127] fits in a *signed* byte,
//     so reinterpreting `a` as s8 is lossless.
// The scalar reference (dot_u8_i8_scalar) defines the semantics; every
// vector kernel must be bit-identical to it (tests/eval/nnue/
// test_dot_product.cpp sweeps them against each other).
namespace nnue::dot
{

    // Reference implementation, and the fallback for platforms without a
    // dedicated kernel. Four independent partial sums so the compiler can
    // keep four vector accumulators when it autovectorizes (a single
    // accumulator serializes on the add's latency).
    template <int N>
    inline std::int32_t dot_u8_i8_scalar(const std::uint8_t *a, const std::int8_t *b)
    {
        static_assert(N > 0);
        std::int32_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        for (; i + 4 <= N; i += 4)
        {
            s0 += static_cast<std::int32_t>(a[i + 0]) * b[i + 0];
            s1 += static_cast<std::int32_t>(a[i + 1]) * b[i + 1];
            s2 += static_cast<std::int32_t>(a[i + 2]) * b[i + 2];
            s3 += static_cast<std::int32_t>(a[i + 3]) * b[i + 3];
        }
        for (; i < N; ++i)
            s0 += static_cast<std::int32_t>(a[i]) * b[i];
        return (s0 + s1) + (s2 + s3);
    }

#if defined(__AVX2__)

    namespace detail
    {
        // acc += dot-product-into-i32-lanes(x_u8, y_i8), 32 bytes at a time.
        inline void dp32_256(__m256i &acc, __m256i x, __m256i y)
        {
#if defined(__AVXVNNI__)
            acc = _mm256_dpbusd_avx_epi32(acc, x, y);
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
            acc = _mm256_dpbusd_epi32(acc, x, y);
#else
            const __m256i pairs = _mm256_maddubs_epi16(x, y); // u8*i8 -> i16 pair sums, no saturation (see contract)
            const __m256i prods = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
            acc = _mm256_add_epi32(acc, prods);
#endif
        }

        inline void dp32_128(__m128i &acc, __m128i x, __m128i y)
        {
#if defined(__AVXVNNI__)
            acc = _mm_dpbusd_avx_epi32(acc, x, y);
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
            acc = _mm_dpbusd_epi32(acc, x, y);
#else
            const __m128i pairs = _mm_maddubs_epi16(x, y);
            const __m128i prods = _mm_madd_epi16(pairs, _mm_set1_epi16(1));
            acc = _mm_add_epi32(acc, prods);
#endif
        }

        inline std::int32_t hsum_i32(__m256i v)
        {
            const __m128i lo = _mm256_castsi256_si128(v);
            const __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
            s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
            return _mm_cvtsi128_si32(s);
        }
    } // namespace detail

    template <int N>
    inline std::int32_t dot_u8_i8_impl(const std::uint8_t *a, const std::int8_t *b)
    {
        static_assert(N > 0);

        // Four independent 256-bit accumulators: vpdpbusd has ~4-5 cycles of
        // latency but 0.5-1 cycle of throughput, so a single accumulator
        // would leave the pipeline mostly idle (measured: this exact
        // serialization was ~26% of total search time before this kernel).
        __m256i acc0 = _mm256_setzero_si256();
        __m256i acc1 = acc0, acc2 = acc0, acc3 = acc0;

        int i = 0;
        for (; i + 128 <= N; i += 128)
        {
            detail::dp32_256(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));
            detail::dp32_256(acc1, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i + 32)),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i + 32)));
            detail::dp32_256(acc2, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i + 64)),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i + 64)));
            detail::dp32_256(acc3, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i + 96)),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i + 96)));
        }
        for (; i + 32 <= N; i += 32)
            detail::dp32_256(acc0, _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + i)),
                             _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + i)));

        std::int32_t sum = detail::hsum_i32(
            _mm256_add_epi32(_mm256_add_epi32(acc0, acc1), _mm256_add_epi32(acc2, acc3)));

        if constexpr (N % 32 >= 16)
        {
            __m128i acc = _mm_setzero_si128();
            detail::dp32_128(acc, _mm_loadu_si128(reinterpret_cast<const __m128i *>(a + i)),
                             _mm_loadu_si128(reinterpret_cast<const __m128i *>(b + i)));
            __m128i s = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
            s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
            sum += _mm_cvtsi128_si32(s);
            i += 16;
        }

        for (; i < N; ++i)
            sum += static_cast<std::int32_t>(a[i]) * b[i];
        return sum;
    }

#elif defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

    template <int N>
    inline std::int32_t dot_u8_i8_impl(const std::uint8_t *a, const std::int8_t *b)
    {
        static_assert(N > 0);

        // a is in [0, 127] so it is losslessly reinterpretable as signed
        // bytes, and sdot (s8 x s8) computes the exact same sum.
        const std::int8_t *as = reinterpret_cast<const std::int8_t *>(a);

        int32x4_t acc0 = vdupq_n_s32(0);
        int32x4_t acc1 = acc0, acc2 = acc0, acc3 = acc0;

        int i = 0;
        for (; i + 64 <= N; i += 64)
        {
            acc0 = vdotq_s32(acc0, vld1q_s8(as + i), vld1q_s8(b + i));
            acc1 = vdotq_s32(acc1, vld1q_s8(as + i + 16), vld1q_s8(b + i + 16));
            acc2 = vdotq_s32(acc2, vld1q_s8(as + i + 32), vld1q_s8(b + i + 32));
            acc3 = vdotq_s32(acc3, vld1q_s8(as + i + 48), vld1q_s8(b + i + 48));
        }
        for (; i + 16 <= N; i += 16)
            acc0 = vdotq_s32(acc0, vld1q_s8(as + i), vld1q_s8(b + i));

        std::int32_t sum = vaddvq_s32(vaddq_s32(vaddq_s32(acc0, acc1), vaddq_s32(acc2, acc3)));

        for (; i < N; ++i)
            sum += static_cast<std::int32_t>(a[i]) * b[i];
        return sum;
    }

#else

    template <int N>
    inline std::int32_t dot_u8_i8_impl(const std::uint8_t *a, const std::int8_t *b)
    {
        return dot_u8_i8_scalar<N>(a, b);
    }

#endif

    template <int N>
    inline std::int32_t dot_u8_i8(const std::uint8_t *a, const std::int8_t *b)
    {
#ifndef NDEBUG
        for (int i = 0; i < N; ++i)
            assert(a[i] <= 127 && "dot_u8_i8: activation out of [0, 127] contract");
#endif
        return dot_u8_i8_impl<N>(a, b);
    }

} // namespace nnue::dot
