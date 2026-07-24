#pragma once

#include <cstddef>
#include <cstdint>
#include <experimental/simd>

namespace stdx = std::experimental;

namespace simd
{
    using int16_v = stdx::native_simd<std::int16_t>;
    using int32_v = stdx::native_simd<std::int32_t>;

    constexpr std::size_t SimdSize16 = int16_v::size();
    constexpr std::size_t SimdSize32 = int32_v::size();
}
