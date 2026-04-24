#pragma once

#ifdef TEXEL_TUNING

#include <array>

struct EvalFeatures
{
    // ===== Pawn structure =====
    double doubled_files = 0.0;
    double isolated_files = 0.0;

    std::array<double, 8> passed_mg{};
    std::array<double, 8> passed_eg{};

    // ===== King safety =====
    double open_files_near_king = 0.0;
    double semi_open_files_near_king = 0.0;

    double heavy_on_open = 0.0;
    double heavy_on_semi_open = 0.0;

    // ===== Bishop pair =====
    double bishop_pair_mg = 0.0;
    double bishop_pair_eg = 0.0;

    // ===== Mobility =====
    std::array<double, 9> knight_mob{};
    std::array<double, 14> bishop_mob{};
    std::array<double, 15> rook_mob{};
    std::array<double, 28> queen_mob{};

    // ===== Mop-up =====
    double king_dist_center = 0.0;
    double king_closeness = 0.0;

    // ===== Piec-Values =====
    std::array<double, constants::PieceTypeCount> material{};
};

#endif