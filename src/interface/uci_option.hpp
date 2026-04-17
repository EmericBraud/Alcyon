#pragma once

#ifdef SPSA_TUNING

#include <string>
#include <string_view>
#include <istream>
#include <algorithm>
#include <type_traits>
#include <concepts>
#include <charconv>
#include <cmath>
#include <cstdlib>
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

        if (ec == std::errc{} && ptr == value.data() + value.size())
        {
            temp_val = std::clamp(temp_val, min_value, max_value);

            if constexpr (std::is_floating_point_v<T>)
                *option_value = static_cast<T>(temp_val) / static_cast<T>(100.0);
            else
                *option_value = static_cast<T>(temp_val);

            logs::debug << option_name << " set to " << *option_value << std::endl;
            return true;
        }

        char *end_ptr = nullptr;
        const double parsed = std::strtod(value.c_str(), &end_ptr);
        if (end_ptr == value.c_str() || (end_ptr && *end_ptr != '\0'))
        {
            logs::uci << "info string error: cannot set option " << option_name << " to invalid value " << value << std::endl;
            return true;
        }

        if constexpr (std::is_floating_point_v<T>)
        {
            const T min_real = static_cast<T>(min_value) / static_cast<T>(100.0);
            const T max_real = static_cast<T>(max_value) / static_cast<T>(100.0);
            const T clamped = std::clamp(static_cast<T>(parsed), min_real, max_real);
            *option_value = clamped;
        }
        else
        {
            if (std::floor(parsed) != parsed)
            {
                logs::uci << "info string error: cannot set option " << option_name << " to non-integer value " << value << std::endl;
                return true;
            }

            const long long clamped = std::clamp(static_cast<long long>(parsed), min_value, max_value);
            *option_value = static_cast<T>(clamped);
        }
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