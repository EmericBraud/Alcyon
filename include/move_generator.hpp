#pragma once

#include <array>
#include <vector>
#include <unordered_map>

#include "board.hpp"
#include "move.hpp"

class MoveGenerator
{
    std::array<std::unordered_map<U64, std::vector<Move>>, N_PIECES_TYPE_HALF> move_tables;
};