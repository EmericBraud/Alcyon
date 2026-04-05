#pragma once

#ifdef SPSA_TUNING

#include <string>
#include <string_view>
#include <istream>
#include <algorithm>
#include <type_traits>
#include <concepts>
#include <charconv>
#include <stdexcept>
#include <limits>

#include "common/logger.hpp"

template <typename T>
concept Integral = std::is_arithmetic_v<T> &&
                   !std::is_same_v<T, bool>;

template <Integral T>
class UCIOption
{
    T *option_value;
    std::string_view option_name;
    long long min_value;
    long long max_value;

    static constexpr long long default_min()
    {
        if constexpr (std::is_floating_point_v<T>)
            return -100000;
        return static_cast<long long>(std::numeric_limits<T>::lowest());
    }

    static constexpr long long default_max()
    {
        if constexpr (std::is_floating_point_v<T>)
            return 100000;
        return static_cast<long long>(std::numeric_limits<T>::max());
    }

public:
    UCIOption(T *_option, std::string_view _option_name, long long _min = default_min(), long long _max = default_max())
        : option_value(_option), option_name(_option_name), min_value(_min), max_value(_max)
    {
        if (min_value > max_value)
            std::swap(min_value, max_value);
    }

    bool parse_input(std::istringstream &input)
    {
        std::string word, name, value;
        while (input >> word)
        {
            if (word == "name")
            {
                while (input >> word && word != "value")
                    name += word;
            }
            if (word == "value")
            {
                while (input >> word)
                    value += word;
            }
        }
        if (name != option_name)
            return false;

        long long temp_val{};
        auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), temp_val);

        if (ec == std::errc::invalid_argument)
            throw std::invalid_argument("Not a number");

        if (ec == std::errc::result_out_of_range)
            throw std::out_of_range("Number out of range");

        if (ptr != value.data() + value.size())
            throw std::invalid_argument("Trailing characters");

        temp_val = std::clamp(temp_val, min_value, max_value);

        if constexpr (std::is_floating_point_v<T>)
            *option_value = static_cast<T>(temp_val) / static_cast<T>(100.0);
        else
            *option_value = static_cast<T>(temp_val);
        logs::debug << option_name << " set to " << *option_value << std::endl;
        return true;
    }

    void init_print()
    {
        if constexpr (std::is_floating_point_v<T>)
            logs::uci << "option name " << option_name
                      << " type spin default " << static_cast<int>(*option_value * 100)
                      << " min " << min_value
                      << " max " << max_value
                      << std::endl;
        else if constexpr (std::is_integral_v<T>)
            logs::uci << "option name " << option_name
                      << " type spin default " << *option_value
                      << " min " << min_value
                      << " max " << max_value
                      << std::endl;
        else
            logs::error << "Unsupported option type for " << option_name << std::endl;
    }

    std::string_view get_name() const
    {
        return option_name;
    }
};

#endif