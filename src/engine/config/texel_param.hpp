#pragma once

#ifdef TEXEL_TUNING

struct TunableParam
{
    double value;
    const double learning_rate_scale;
    TunableParam(double _initial_value) : value(_initial_value), learning_rate_scale(1.0 / std::max(1.0, std::abs(_initial_value)))
    {
    }

    operator double() const { return value; };
};
#endif