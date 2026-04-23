/// This code is responsible for the texel tuning of the engine.
/// It should not be used nor compiled in other situations !
#pragma once

#ifdef TEXEL_TUNING

#include <string>
#include <string_view>
#include <fstream>

#include "common/file.hpp"
struct TexelSample
{
    std::string fen;
    double result; // 1.0, 0.5, 0.0
    int eval;
};

std::optional<TexelSample> parse_texel_line(const std::string &line)
{
    std::istringstream iss(line);

    std::string fen1, fen2, fen3, fen4, fen5, fen6;
    std::string result_token;
    int eval = 0;

    if (!(iss >> fen1 >> fen2 >> fen3 >> fen4 >> fen5 >> fen6 >> result_token >> eval))
    {
        return std::nullopt;
    }

    std::string fen = fen1 + " " + fen2 + " " + fen3 + " " + fen4 + " " + fen5 + " " + fen6;

    if (result_token.size() < 3 || result_token.front() != '[' || result_token.back() != ']')
    {
        return std::nullopt;
    }

    double result = std::stod(result_token.substr(1, result_token.size() - 2));

    return TexelSample{fen, result, eval};
}

class TexelTuner
{
    void start_tuning()
    {
        std::ifstream epd_file(file::get_data_path("tuning/E12.52-1M-D12-Resolved"));

        if (!epd_file.is_open())
        {
            throw std::runtime_error("Cannot open tuning file");
        }
        std::string line;
        while (std::getline(epd_file, line))
        {
            auto r = parse_texel_line(line);
            if (!r.has_value())
                continue;
            TexelSample sample = r.value();
                }
    }
};
#endif