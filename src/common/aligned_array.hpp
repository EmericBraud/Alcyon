#pragma once

#include <array>
#include <cstddef>
#include <new>

// Wraps std::array<T, N> to force its alignment to the cache-line /
// destructive-interference size: sizeof stays a multiple of alignof (unlike
// a bare `using X alignas(N) = std::array<...>` alias, which GCC rejects for
// array-of-X use -- "size of array element is not a multiple of its
// alignment"), and since it inherits std::array's interface unchanged
// (operator[], .data(), aggregate-init from a plain std::array value, ...),
// every existing call site keeps working as-is.
template <typename T, std::size_t N>
struct alignas(std::hardware_destructive_interference_size) AlignedArray : std::array<T, N>
{
};
