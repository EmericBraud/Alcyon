#pragma once


#include "board.hpp"

class MoveGenerator
{
    std::array<std::unordered_map<U64, std::vector<Move>>, N_PIECES_TYPE_HALF> move_tables;
};