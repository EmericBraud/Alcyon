// Bit-exactness of the vector dot kernels against the scalar reference:
// dot_u8_i8 must return exactly dot_u8_i8_scalar for every input honoring
// the [0,127] x [-128,127] contract (see dot_product.hpp). Sizes cover the
// network's real layer widths (1024, 16, 64, 80) plus odd/tail cases.
#include "gtest/gtest.h"
#include "engine/eval/nnue/dot_product.hpp"

#include <array>
#include <cstdint>
#include <random>

namespace
{

    template <int N>
    void expect_matches_scalar(const std::array<std::uint8_t, N> &a, const std::array<std::int8_t, N> &b)
    {
        ASSERT_EQ((nnue::dot::dot_u8_i8<N>(a.data(), b.data())),
                  (nnue::dot::dot_u8_i8_scalar<N>(a.data(), b.data())));
    }

    template <int N>
    void run_random_sweep(unsigned seed, int iterations)
    {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> act(0, 127);
        std::uniform_int_distribution<int> wgt(-128, 127);

        std::array<std::uint8_t, N> a;
        std::array<std::int8_t, N> b;
        for (int it = 0; it < iterations; ++it)
        {
            for (auto &x : a)
                x = static_cast<std::uint8_t>(act(rng));
            for (auto &x : b)
                x = static_cast<std::int8_t>(wgt(rng));
            expect_matches_scalar<N>(a, b);
        }
    }

    template <int N>
    void run_extremes()
    {
        std::array<std::uint8_t, N> a;
        std::array<std::int8_t, N> b;

        // Worst positive case: 127 * 127 everywhere.
        a.fill(127);
        b.fill(127);
        expect_matches_scalar<N>(a, b);
        ASSERT_EQ((nnue::dot::dot_u8_i8<N>(a.data(), b.data())), 127 * 127 * N);

        // Worst negative case: 127 * -128 everywhere -- the case that would
        // saturate a vpmaddubsw-based kernel if the pair-sum bound were wrong.
        b.fill(-128);
        expect_matches_scalar<N>(a, b);
        ASSERT_EQ((nnue::dot::dot_u8_i8<N>(a.data(), b.data())), 127 * -128 * N);

        // Alternating extremes stress the i16 pair sums of the AVX2 path.
        for (int i = 0; i < N; ++i)
            b[i] = (i % 2 == 0) ? 127 : -128;
        expect_matches_scalar<N>(a, b);

        a.fill(0);
        expect_matches_scalar<N>(a, b);
        ASSERT_EQ((nnue::dot::dot_u8_i8<N>(a.data(), b.data())), 0);
    }

} // namespace

TEST(DotProductTest, ScalarReferenceKnownValues)
{
    const std::array<std::uint8_t, 4> a = {1, 2, 3, 4};
    const std::array<std::int8_t, 4> b = {10, -10, 5, -5};
    ASSERT_EQ((nnue::dot::dot_u8_i8_scalar<4>(a.data(), b.data())), 10 - 20 + 15 - 20);
}

TEST(DotProductTest, MatchesScalarOnNetworkSizes)
{
    run_random_sweep<1024>(0xC0FFEE, 200); // L1: NAccumulator inputs
    run_random_sweep<16>(0xBEEF, 2000);    // L2 input / output-layer split A
    run_random_sweep<64>(0xF00D, 2000);    // output-layer split B
    run_random_sweep<80>(0xCAFE, 2000);    // output layer, unsplit
}

TEST(DotProductTest, MatchesScalarOnTailSizes)
{
    // Exercise every code path: 128-byte unrolled loop, 32-byte loop,
    // 16-byte step, scalar remainder.
    run_random_sweep<256>(1, 500);
    run_random_sweep<160>(2, 500);
    run_random_sweep<48>(3, 1000);
    run_random_sweep<32>(4, 1000);
    run_random_sweep<17>(5, 1000);
    run_random_sweep<8>(6, 1000);
    run_random_sweep<3>(7, 1000);
    run_random_sweep<1>(8, 1000);
}

TEST(DotProductTest, ExtremeValues)
{
    run_extremes<1024>();
    run_extremes<80>();
    run_extremes<64>();
    run_extremes<16>();
    run_extremes<17>();
    run_extremes<1>();
}
