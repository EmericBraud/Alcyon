#pragma once

// Scoped incremental update helpers for Full_Threats (see full_threats_encoder.hpp).
//
// A single move can, in principle, change any threat feature whose attacker is a
// sliding piece (bishop/rook/queen) anywhere on the board -- moving a piece off or
// onto a square can open or close *any* ray, not just one touching the moved
// squares. A truly minimal per-move diff would require walking rays outward from
// every occupancy change to find the "next" blocker in each of the 8 directions.
// Instead we take the pragmatic middle ground explicitly allowed by the spec:
// recompute the outgoing threats of every slider on the board (bounded by the
// number of bishops/rooks/queens in play, typically small and always <= the full
// piece count), plus the full attacker/defender contribution of the handful of
// squares whose occupant actually changed (from/to/en-passant-capture). This is
// strictly cheaper than a full-board recompute (which also re-walks every pawn,
// knight and king from scratch) while staying provably correct: every feature
// that *can* change is covered by one of the two scans below.

#include <array>
#include <vector>

#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/eval/nnue/full_threats_encoder.hpp"

namespace nnue::threats
{
    // Attack bitboard (intersected with real occupancy, i.e. only squares that
    // hold a piece) for a single piece -- mirrors fill_features's per-piece
    // attacker logic exactly, so it stays a single source of truth.
    inline U64 piece_attack_targets(const Board &board, Piece piece_type, Color color, int from)
    {
        const U64 all_pieces = board.occupancies[NO_COLOR];

        if (piece_type == PAWN)
        {
            U64 attacks = pawn_attacks_no_blockers(color, from) & all_pieces;
            const int push = (color == WHITE) ? 8 : -8;
            const int push_sq = from + push;
            if (push_sq >= 0 && push_sq < 64 && (all_pieces & (1ULL << push_sq)))
                attacks |= (1ULL << push_sq);
            return attacks;
        }

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
            attacks = MoveGen::generate_bishop_moves(from, all_pieces);
            break;
        case ROOK:
            attacks = MoveGen::generate_rook_moves(from, all_pieces);
            break;
        case QUEEN:
            attacks = MoveGen::generate_rook_moves(from, all_pieces) | MoveGen::generate_bishop_moves(from, all_pieces);
            break;
        default:
            attacks = 0ULL;
        }
        return attacks & all_pieces;
    }

    // Appends every threat feature where `sq` is the attacker's square.
    template <Color perspective>
    inline void collect_attacker_features_from(const Board &board, int ksq, int sq, std::vector<int> &out)
    {
        const Piece pt = board.get_p(sq);
        if (pt == NO_PIECE || pt == KING)
            return;

        const Color color = board.get_c(sq);
        U64 targets = piece_attack_targets(board, pt, color, sq);
        while (targets)
        {
            const int to = cpu::pop_lsb(targets);
            const Piece attkd_type = board.get_p(to);
            const Color attkd_color = board.get_c(to);
            if (attkd_type == NO_PIECE)
                continue;

            const int idx = threat_index<perspective>(pt, color, sq, to, attkd_type, attkd_color, ksq);
            if (idx >= 0)
                out.push_back(idx);
        }
    }

    // Appends every threat feature where `sq` is the defender's (attacked) square:
    // scans every non-king piece on the board and checks whether it attacks `sq`.
    // Only called for the handful of squares whose occupant changed this move.
    template <Color perspective>
    inline void collect_defender_features_at(const Board &board, int ksq, int sq, std::vector<int> &out)
    {
        const Piece defender_type = board.get_p(sq);
        if (defender_type == NO_PIECE)
            return;
        const Color defender_color = board.get_c(sq);
        const U64 sq_mask = 1ULL << sq;

        for (int c = 0; c < 2; ++c)
        {
            const Color color = static_cast<Color>(c);
            for (int pt = PAWN; pt < KING; ++pt)
            {
                const Piece piece_type = static_cast<Piece>(pt);
                U64 bb = board.pieces_occ[get_piece_index(piece_type, color)];
                while (bb)
                {
                    const int from = cpu::pop_lsb(bb);
                    if (!(piece_attack_targets(board, piece_type, color, from) & sq_mask))
                        continue;

                    const int idx = threat_index<perspective>(piece_type, color, from, sq, defender_type, defender_color, ksq);
                    if (idx >= 0)
                        out.push_back(idx);
                }
            }
        }
    }

    // Full scoped-recompute entry point for one board snapshot (either strictly
    // before or strictly after a non-king move): `touched_squares` are the
    // squares whose occupant changed (from/to/en-passant-capture square).
    template <Color perspective>
    inline void collect_move_scoped_features(const Board &board, const std::vector<int> &touched_squares, std::vector<int> &out)
    {
        const int ksq = board.king_sq[perspective];

        for (int sq : touched_squares)
        {
            collect_attacker_features_from<perspective>(board, ksq, sq, out);
            collect_defender_features_at<perspective>(board, ksq, sq, out);
        }

        constexpr Piece Sliders[3] = {BISHOP, ROOK, QUEEN};
        for (int c = 0; c < 2; ++c)
        {
            const Color color = static_cast<Color>(c);
            for (Piece pt : Sliders)
            {
                U64 bb = board.pieces_occ[get_piece_index(pt, color)];
                while (bb)
                {
                    const int sq = cpu::pop_lsb(bb);
                    collect_attacker_features_from<perspective>(board, ksq, sq, out);
                }
            }
        }
    }
}
