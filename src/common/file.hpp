#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <stdexcept>
#include <array>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#include "constants.hpp"

namespace file
{
    inline std::filesystem::path get_executable_dir()
    {
#ifdef __linux__
        std::error_code ec_linux;
        const std::filesystem::path exe_path = std::filesystem::read_symlink("/proc/self/exe", ec_linux);
        if (!ec_linux && !exe_path.empty())
            return exe_path.parent_path();
#elif defined(__APPLE__)
        char executable_path[PATH_MAX];
        uint32_t size = sizeof(executable_path);
        if (_NSGetExecutablePath(executable_path, &size) == 0)
            return std::filesystem::path(executable_path).parent_path();
#endif
        std::error_code ec;
        const std::filesystem::path cwd = std::filesystem::current_path(ec);
        if (!ec)
            return cwd;
        return std::filesystem::path{"."};
    }

    inline std::filesystem::path get_data_dir()
    {
        if (const char *override_dir = std::getenv("CHESS26_DATA_DIR"))
        {
            if (*override_dir != '\0')
                return std::filesystem::path{override_dir};
        }
        return get_executable_dir() / "data";
    }

    inline std::string get_data_path(std::string_view filename)
    {
        return (get_data_dir() / std::filesystem::path(filename)).string();
    }

    constexpr std::array<std::array<const char *, constants::PieceTypeCount>, 2> PieceImagePath = {
        {// WHITE
         {
             "pieces/white-pawn.png",   // PAWN
             "pieces/white-knight.png", // KNIGHT
             "pieces/white-bishop.png", // BISHOP
             "pieces/white-rook.png",   // ROOK
             "pieces/white-queen.png",  // QUEEN
             "pieces/white-king.png"    // KING
         },
         // BLACK
         {
             "pieces/black-pawn.png",   // PAWN
             "pieces/black-knight.png", // KNIGHT
             "pieces/black-bishop.png", // BISHOP
             "pieces/black-rook.png",   // ROOK
             "pieces/black-queen.png",  // QUEEN
             "pieces/black-king.png"    // KING
         }}};

    constexpr int RookAttacksFileSize = 102400;
    constexpr int BishopAttacksFileSize = 5248;

}
