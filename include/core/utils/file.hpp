#pragma once

#include <string>
#include <stdexcept>
#include <array>

#include "constants.hpp"

namespace core::file
{
    inline std::string get_data_path(std::string_view filename)
    {
#ifndef DATA_PATH
#error "DATA_PATH is undefined"
#endif
        std::string path;
        path.reserve(std::string_view(DATA_PATH).size() + filename.size()); // DATA_PATH is defined by CMake at compile time
        path += DATA_PATH;
        path += filename;
        return path;
    }

    constexpr std::array<std::array<const char *, core::constants::PieceTypeCount>, 2> PieceImagePath = {
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
