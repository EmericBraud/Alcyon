#pragma once

#include <cstdint>
#include <cassert>

enum Color : std::uint8_t
{
    WHITE,
    BLACK,
    NO_COLOR
};

constexpr Color operator!(Color c)
{
    assert(c != NO_COLOR);
    return static_cast<Color>(c ^ 1);
}