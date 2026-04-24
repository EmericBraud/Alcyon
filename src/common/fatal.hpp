#pragma once

#include <cstdlib>
#include <iostream>

#define FATAL(msg)                             \
    do                                         \
    {                                          \
        std::cerr << "FATAL: " << msg          \
                  << " (" << __FILE__          \
                  << ":" << __LINE__ << ")\n"; \
        std::abort();                          \
    } while (0)

//