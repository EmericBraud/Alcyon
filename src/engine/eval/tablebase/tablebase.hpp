#pragma once

#include <string_view>
#include <array>

extern "C"
{
#include "engine/eval/tablebase/fathom/tbprobe.h"
}

constexpr std::array<std::string_view, 0> tablebase_file_path;
class TableBase
{
    TableBase()
    {
        for (auto s : tablebase_file_path)
        {
            tb_init(s);
        }
    }
}