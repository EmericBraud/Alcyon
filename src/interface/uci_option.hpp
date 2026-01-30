#pragma once

#ifdef SPSA_TUNING

#include <string>
#include <string_view>
#include <istream>
#include <type_traits>
#include <concepts>
#include <charconv>
#include <stdexcept>

#include "common/logger.hpp"

template <typename T>
concept Integral = std::is_arithmetic_v<T> &&
                   !std::is_same_v<T, bool>;

template <Integral T>
class UCIOption
{
    T *option_value;
    std::string_view option_name;

public:
    UCIOption(T *_option, std::string_view _option_name) : option_value(_option), option_name(_option_name) {}

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
            logs::uci << "option name " << option_name << " default spin default " << static_cast<int>(*option_value * 100) << std::endl;
        else if constexpr (std::is_integral_v<T>)
            logs::uci << "option name " << option_name << " default spin default " << *option_value << std::endl;
        else
            logs::error << "Unsupported option type for " << option_name << std::endl;
    }

    std::string_view get_name() const
    {
        return option_name;
    }
};

#endif