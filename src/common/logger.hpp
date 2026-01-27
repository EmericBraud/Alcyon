#pragma once

#include <iostream>

namespace logs
{
    class DebugLogger
    {
    public:
#ifndef NDEBUG
        template <typename T>
        DebugLogger &operator<<(const T &v)
        {
            std::cout << v;
            return *this;
        }

        // for std::endl, std::flush
        DebugLogger &operator<<(std::ostream &(*pf)(std::ostream &))
        {
            std::cout << pf;
            return *this;
        }
#else
        template <typename T>
        DebugLogger &operator<<(const T &)
        {
            return *this;
        }

        DebugLogger &operator<<(std::ostream &(*)(std::ostream &))
        {
            return *this;
        }
#endif
    };

    extern DebugLogger debug;

    class UCILogger
    {
    public:
        template <typename T>
        UCILogger &operator<<(const T &v)
        {
            std::cout << v;
            return *this;
        }

        UCILogger &operator<<(std::ostream &(*pf)(std::ostream &))
        {
            std::cout << pf;
            return *this;
        }
    };

    extern UCILogger uci;
    extern UCILogger error;
}
