#pragma once

// Port of nnue-pytorch's Full_Threats feature set, as implemented in
// data_loader/cpp/training_data_loader.cpp (there is no pure-python reference:
// this feature only exists as a training-data-generation C++ extractor) at
// commit 4289208fe20cc6ec8753e5ee14c2f210de783ff0.
//
// Full_Threats has 60,720 real features (NUM_REAL_FEATURES == NUM_INPUTS, no
// virtual/factorizer weights). Each feature represents "piece of type/color X
// attacks (or defends) piece of type/color Y, at this specific relative
// from/to encoding, from this perspective". See threat_index() below for the
// exact index formula, transcribed from the source's `FullThreats::threat_index`.

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include "common/constants.hpp"
#include "common/mask.hpp"
#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"

namespace nnue_v2::threats
{
    // numvalidtargets[piece] = 2 * (number of attacked-piece-types this attacker
    // type can threaten), piece indexed as 2*type + color (White=0, Black=1).
    // Pawn=6, Knight=10, Bishop=8, Rook=8, Queen=10, King=0 (kings are never
    // tracked as attackers), duplicated for both colors.
    constexpr int numvalidtargets[12] = {6, 6, 10, 10, 8, 8, 8, 8, 10, 10, 0, 0};

    // map[attacker_type][defender_type] -> "target group" index, or -1 if that
    // attacker/defender type combination is never tracked. Types ordered
    // Pawn, Knight, Bishop, Rook, Queen, King (matching this engine's Piece enum).
    // clang-format off
    constexpr int threat_map[6][6] = {
        {0, 1, -1, 2, -1, -1}, // Pawn attacks: Pawn, Knight, Rook
        {0, 1, 2, 3, 4, -1},   // Knight attacks: Pawn, Knight, Bishop, Rook, Queen
        {0, 1, 2, 3, -1, -1},  // Bishop attacks: Pawn, Knight, Bishop, Rook
        {0, 1, 2, 3, -1, -1},  // Rook attacks: Pawn, Knight, Bishop, Rook
        {0, 1, 2, 3, 4, -1},   // Queen attacks: Pawn, Knight, Bishop, Rook, Queen
        {-1, -1, -1, -1, -1, -1}, // King: never an attacker
    };
    // clang-format on

    constexpr int NUM_INPUTS = 60720;

    struct ThreatOffsetTable
    {
        // table[piece][from] (from in 0..63) = cumulative pseudo-attack-count
        //   offset for squares before `from`, for that attacker piece.
        // table[piece][64] = total pseudo-attack-count sum over all 64 squares
        //   (used as the per-"target group" stride).
        // table[piece][65] = cumulative feature offset before this attacker
        //   piece's whole block.
        std::array<std::array<int, 66>, 12> table{};
        int total_features = 0;
    };

    inline U64 pseudo_attacks_no_blockers(Piece type, int sq)
    {
        switch (type)
        {
        case KNIGHT:
            return MoveGen::KnightAttacks[sq];
        case KING:
            return MoveGen::KingAttacks[sq];
        case BISHOP:
            return MoveGen::generate_bishop_moves(sq, 0ULL);
        case ROOK:
            return MoveGen::generate_rook_moves(sq, 0ULL);
        case QUEEN:
            return MoveGen::generate_rook_moves(sq, 0ULL) | MoveGen::generate_bishop_moves(sq, 0ULL);
        default:
            return 0ULL;
        }
    }

    inline U64 pawn_attacks_no_blockers(Color color, int sq)
    {
        return color == WHITE ? MoveGen::PawnAttacksWhite[sq] : MoveGen::PawnAttacksBlack[sq];
    }

    inline const ThreatOffsetTable &offset_table()
    {
        static const ThreatOffsetTable instance = []
        {
            ThreatOffsetTable t{};
            int piece_offset = 0;

            for (int c = 0; c < 2; ++c)
            {
                for (int pt = 0; pt < 6; ++pt)
                {
                    const int piece = 2 * pt + c;
                    const Piece piece_type = static_cast<Piece>(pt);
                    const Color piece_color = static_cast<Color>(c);

                    t.table[piece][65] = piece_offset;
                    int square_offset = 0;

                    for (int from = 0; from < 64; ++from)
                    {
                        t.table[piece][from] = square_offset;

                        if (piece_type != PAWN)
                        {
                            const U64 attacks = pseudo_attacks_no_blockers(piece_type, from);
                            square_offset += std::popcount(attacks);
                        }
                        else if (from >= 8 && from < 56) // a2..h7
                        {
                            U64 attacks = pawn_attacks_no_blockers(piece_color, from);
                            const int push = (piece_color == WHITE) ? 8 : -8;
                            attacks |= (1ULL << (from + push));
                            square_offset += std::popcount(attacks);
                        }
                    }

                    t.table[piece][64] = square_offset;
                    piece_offset += numvalidtargets[piece] * square_offset;
                }
            }

            t.total_features = piece_offset;
            return t;
        }();

        return instance;
    }

    // Precomputed once via offset_table(); NUM_INPUTS should match its total
    // (verified at runtime by test_full_threats_encoder.cpp, since the table
    // is built from runtime attack tables and can't be a constexpr check).

    // Returns the oriented-square XOR mask for this perspective and king
    // square: mirrors the file so the king lands on a "canonical" half of the
    // board (a-d), and flips the whole board vertically for Black's POV.
    template <Color perspective>
    inline int orient_mask(int ksq)
    {
        const int kfile = ksq % 8;
        if constexpr (perspective == WHITE)
            return (kfile < 4) ? 0 : 7; // a1 : h1
        else
            return (kfile < 4) ? 56 : 63; // a8 : h8
    }

    // Returns the Full_Threats feature index for attacker (attkr_type,
    // attkr_color) at `from`, attacking (attkd_type, attkd_color) at `to`, or
    // -1 if this attacker/defender pair isn't tracked (or is a redundant
    // symmetric duplicate that's pruned to avoid double counting).
    template <Color perspective>
    inline int threat_index(Piece attkr_type, Color attkr_color, int from,
                             int to, Piece attkd_type, Color attkd_color, int ksq)
    {
        const bool enemy = (attkr_color != attkd_color);

        const int orient = orient_mask<perspective>(ksq);
        from ^= orient;
        to ^= orient;

        if constexpr (perspective == BLACK)
        {
            attkr_color = !attkr_color;
            attkd_color = !attkd_color;
        }

        const int group = threat_map[attkr_type][attkd_type];
        if (group < 0)
            return -1;
        if (attkr_type == attkd_type && (enemy || attkr_type != PAWN) && from < to)
            return -1;

        U64 attacks;
        if (attkr_type == PAWN)
        {
            attacks = pawn_attacks_no_blockers(attkr_color, from);
            const int push = (attkr_color == WHITE) ? 8 : -8;
            attacks |= (1ULL << (from + push));
        }
        else
        {
            attacks = pseudo_attacks_no_blockers(attkr_type, from);
        }

        const auto &t = offset_table();
        const int attkr_piece = 2 * static_cast<int>(attkr_type) + static_cast<int>(attkr_color);
        const int nvt = numvalidtargets[attkr_piece];

        const U64 below_to_mask = (to == 0) ? 0ULL : ((to >= 64) ? ~0ULL : ((1ULL << to) - 1));
        const int position_in_attacks = std::popcount(attacks & below_to_mask);

        return t.table[attkr_piece][65]
               + (static_cast<int>(attkd_color) * (nvt / 2) + group) * t.table[attkr_piece][64]
               + t.table[attkr_piece][from]
               + position_in_attacks;
    }

    // Appends all active Full_Threats feature indices (from `perspective`'s
    // point of view) for the given board to `out`.
    template <Color perspective>
    inline void fill_features(const Board &board, std::vector<int> &out)
    {
        const int ksq = board.king_sq[perspective];
        const U64 all_pieces = board.occupancies[NO_COLOR];

        for (int c = 0; c < 2; ++c)
        {
            const Color color = static_cast<Color>(c);

            for (int pt = PAWN; pt <= KING; ++pt)
            {
                const Piece piece_type = static_cast<Piece>(pt);
                U64 bb = board.pieces_occ[get_piece_index(piece_type, color)];

                if (piece_type == PAWN)
                {
                    U64 attackers = bb;
                    while (attackers)
                    {
                        const int from = cpu::pop_lsb(attackers);
                        U64 attacks = pawn_attacks_no_blockers(color, from) & all_pieces;
                        const int push = (color == WHITE) ? 8 : -8;
                        const int push_sq = from + push;
                        if (push_sq >= 0 && push_sq < 64 && (all_pieces & (1ULL << push_sq)))
                            attacks |= (1ULL << push_sq);

                        U64 targets = attacks;
                        while (targets)
                        {
                            const int to = cpu::pop_lsb(targets);
                            const Piece attkd_type = board.get_p(to);
                            const Color attkd_color = board.get_c(to);
                            if (attkd_type == NO_PIECE)
                                continue;

                            const int idx = threat_index<perspective>(
                                piece_type, color, from, to, attkd_type, attkd_color, ksq);
                            if (idx >= 0)
                                out.push_back(idx);
                        }
                    }
                }
                else
                {
                    U64 attackers = bb;
                    while (attackers)
                    {
                        const int from = cpu::pop_lsb(attackers);
                        const U64 occ = all_pieces;
                        U64 attacks;
                        switch (piece_type)
                        {
                        case KNIGHT:
                            attacks = MoveGen::KnightAttacks[from];
                            break;
                        case KING:
                            attacks = MoveGen::KingAttacks[from];
                            break;
                        case BISHOP:
                            attacks = MoveGen::generate_bishop_moves(from, occ);
                            break;
                        case ROOK:
                            attacks = MoveGen::generate_rook_moves(from, occ);
                            break;
                        case QUEEN:
                            attacks = MoveGen::generate_rook_moves(from, occ) | MoveGen::generate_bishop_moves(from, occ);
                            break;
                        default:
                            attacks = 0ULL;
                        }
                        attacks &= all_pieces;

                        U64 targets = attacks;
                        while (targets)
                        {
                            const int to = cpu::pop_lsb(targets);
                            const Piece attkd_type = board.get_p(to);
                            const Color attkd_color = board.get_c(to);
                            if (attkd_type == NO_PIECE)
                                continue;

                            const int idx = threat_index<perspective>(
                                piece_type, color, from, to, attkd_type, attkd_color, ksq);
                            if (idx >= 0)
                                out.push_back(idx);
                        }
                    }
                }
            }
        }
    }
}
