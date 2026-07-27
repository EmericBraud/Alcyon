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

#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/eval/nnue/full_threats_encoder.hpp"
#include "engine/eval/nnue/fixed_int_list.hpp"

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
    inline void collect_attacker_features_from(const Board &board, int ksq, int sq, FixedIntList<MAX_THREAT_FEATURES> &out)
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

            // push_back_if : "idx valide pour cette perspective" est
            // data-dépendant (~imprévisible), voir filter_threats_diff.
            const int idx = threat_index<perspective>(pt, color, sq, to, attkd_type, attkd_color, ksq);
            out.push_back_if(idx, idx >= 0);
        }
    }

    // Appends every threat feature where `sq` is the defender's (attacked) square.
    // Only called for the handful of squares whose occupant changed this move.
    // Derived in O(1) via MoveGen::attackers_to() -- see collect_defender_features_at_both,
    // whose reverse-push handling for Full_Threats' pawn pseudo-attack this mirrors.
    //
    // `touched_mask` excludes attackers that are themselves touched squares:
    // those already get their outgoing (attacker-role) features emitted by
    // collect_attacker_features_from() in the same touched_squares loop, which
    // for an attacker A also touched would independently discover the exact
    // same (A -> sq) pair -- skipping it here avoids emitting that idx twice.
    template <Color perspective>
    inline void collect_defender_features_at(const Board &board, int ksq, int sq, U64 touched_mask, FixedIntList<MAX_THREAT_FEATURES> &out)
    {
        const Piece defender_type = board.get_p(sq);
        if (defender_type == NO_PIECE)
            return;
        const Color defender_color = board.get_c(sq);
        const U64 occ = board.occupancies[NO_COLOR];

        U64 attackers = MoveGen::attackers_to(sq, occ, board) & ~touched_mask;
        while (attackers)
        {
            const int from = cpu::pop_lsb(attackers);
            const Piece attkr_type = board.get_p(from);
            const Color attkr_color = board.get_c(from);
            const int idx = threat_index<perspective>(attkr_type, attkr_color, from, sq, defender_type, defender_color, ksq);
            out.push_back_if(idx, idx >= 0);
        }

        if (sq >= 8)
        {
            const int from = sq - 8;
            if (board.get_p(from) == PAWN && board.get_c(from) == WHITE)
            {
                const int idx = threat_index<perspective>(PAWN, WHITE, from, sq, defender_type, defender_color, ksq);
                if (idx >= 0)
                    out.push_back(idx);
            }
        }
        if (sq < 56)
        {
            const int from = sq + 8;
            if (board.get_p(from) == PAWN && board.get_c(from) == BLACK)
            {
                const int idx = threat_index<perspective>(PAWN, BLACK, from, sq, defender_type, defender_color, ksq);
                if (idx >= 0)
                    out.push_back(idx);
            }
        }
    }

    // Combined-perspective variants of the two collectors above: the attacker/
    // defender scan itself (piece_attack_targets, the magic-bitboard lookups)
    // does not depend on `perspective` at all -- only the final threat_index
    // lookup does (via ksq and the White/Black color flip). Calling the
    // templated per-perspective collectors once for WHITE and once for BLACK
    // therefore redoes the same attack-generation work twice; these variants
    // do it once and derive both perspectives' indices from the same scan.
    inline void collect_attacker_features_from_both(
        const Board &board, int wksq, int bksq, int sq,
        FixedIntList<MAX_THREAT_FEATURES> &out_white, FixedIntList<MAX_THREAT_FEATURES> &out_black)
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

            // push_back_if : "idx valide pour cette perspective" est
            // data-dépendant (~imprévisible), voir filter_threats_diff.
            const int idx_w = threat_index<WHITE>(pt, color, sq, to, attkd_type, attkd_color, wksq);
            out_white.push_back_if(idx_w, idx_w >= 0);

            const int idx_b = threat_index<BLACK>(pt, color, sq, to, attkd_type, attkd_color, bksq);
            out_black.push_back_if(idx_b, idx_b >= 0);
        }
    }

    // Emits both perspectives' threat_index for one (attacker at `from`,
    // defender at `sq`) pair -- shared by the two call sites below.
    inline void emit_defender_pair(
        Piece attkr_type, Color attkr_color, int from, int sq, Piece defender_type, Color defender_color,
        int wksq, int bksq, FixedIntList<MAX_THREAT_FEATURES> &out_white, FixedIntList<MAX_THREAT_FEATURES> &out_black)
    {
        const int idx_w = threat_index<WHITE>(attkr_type, attkr_color, from, sq, defender_type, defender_color, wksq);
        out_white.push_back_if(idx_w, idx_w >= 0);

        const int idx_b = threat_index<BLACK>(attkr_type, attkr_color, from, sq, defender_type, defender_color, bksq);
        out_black.push_back_if(idx_b, idx_b >= 0);
    }

    // Same result as scanning every non-king piece on the board and checking
    // whether it attacks `sq` (see the piece-by-piece version this replaced),
    // but derived in O(1) via MoveGen::attackers_to() -- the same reverse
    // magic-bitboard lookup SEE uses (see.cpp) -- instead of generating each
    // of the <= 30 pieces' full attack set to test membership. The one case
    // attackers_to() doesn't cover is Full_Threats' pawn "push" pseudo-attack
    // (a pawn "attacks" the empty-in-real-chess-terms square directly ahead
    // of it, per double_feature_transform's convention): handled separately
    // below via two O(1) reverse-push checks.
    // `touched_mask` excludes attackers that are themselves touched squares --
    // see collect_defender_features_at()'s comment for why.
    inline void collect_defender_features_at_both(
        const Board &board, int wksq, int bksq, int sq, U64 touched_mask,
        FixedIntList<MAX_THREAT_FEATURES> &out_white, FixedIntList<MAX_THREAT_FEATURES> &out_black)
    {
        const Piece defender_type = board.get_p(sq);
        if (defender_type == NO_PIECE)
            return;
        const Color defender_color = board.get_c(sq);
        const U64 occ = board.occupancies[NO_COLOR];

        U64 attackers = MoveGen::attackers_to(sq, occ, board) & ~touched_mask;
        while (attackers)
        {
            const int from = cpu::pop_lsb(attackers);
            const Piece attkr_type = board.get_p(from);
            const Color attkr_color = board.get_c(from);
            emit_defender_pair(attkr_type, attkr_color, from, sq, defender_type, defender_color, wksq, bksq, out_white, out_black);
        }

        // Reverse-push case: a WHITE pawn at sq-8 (or BLACK pawn at sq+8)
        // "attacks" sq via its forward push, which attackers_to() doesn't
        // model (real chess attacks never include the push square).
        if (sq >= 8)
        {
            const int from = sq - 8;
            if (board.get_p(from) == PAWN && board.get_c(from) == WHITE)
                emit_defender_pair(PAWN, WHITE, from, sq, defender_type, defender_color, wksq, bksq, out_white, out_black);
        }
        if (sq < 56)
        {
            const int from = sq + 8;
            if (board.get_p(from) == PAWN && board.get_c(from) == BLACK)
                emit_defender_pair(PAWN, BLACK, from, sq, defender_type, defender_color, wksq, bksq, out_white, out_black);
        }
    }

    // Combined-perspective entry point: same scoped-recompute scheme as
    // collect_move_scoped_features() below, but shares the magic-bitboard
    // attacker/defender scan between both perspectives instead of repeating
    // it once per perspective (see the *_both() helpers above).
    inline void collect_move_scoped_features_both(
        const Board &board, const FixedIntList<MAX_TOUCHED_SQUARES> &touched_squares,
        FixedIntList<MAX_THREAT_FEATURES> &out_white, FixedIntList<MAX_THREAT_FEATURES> &out_black)
    {
        const int wksq = board.king_sq[WHITE];
        const int bksq = board.king_sq[BLACK];
        const U64 occ = board.occupancies[NO_COLOR];
        const U64 rooks_queens = board.pieces_occ[get_piece_index(ROOK, WHITE)] | board.pieces_occ[get_piece_index(ROOK, BLACK)] |
                                 board.pieces_occ[get_piece_index(QUEEN, WHITE)] | board.pieces_occ[get_piece_index(QUEEN, BLACK)];
        const U64 bishops_queens = board.pieces_occ[get_piece_index(BISHOP, WHITE)] | board.pieces_occ[get_piece_index(BISHOP, BLACK)] |
                                   board.pieces_occ[get_piece_index(QUEEN, WHITE)] | board.pieces_occ[get_piece_index(QUEEN, BLACK)];

        // A touched square that is itself a slider is already covered by the
        // touched_squares loop below (collect_attacker_features_from_both) --
        // masked out of slider_candidates so the final while() loop doesn't
        // re-emit the same square's features a second time. Measured ~26.6%
        // of slider_candidates entries overlapped touched_squares before this.
        //
        // touched_mask is computed upfront (not incrementally inside the loop
        // below) because collect_defender_features_at_both() needs the full
        // set of touched squares regardless of iteration order: an attacker A
        // touched later in the loop still needs to be excluded when scanning
        // an earlier touched square's defenders.
        U64 touched_mask = 0ULL;
        for (int sq : touched_squares)
            touched_mask |= (1ULL << sq);

        U64 slider_candidates = 0ULL;
        for (int sq : touched_squares)
        {
            collect_attacker_features_from_both(board, wksq, bksq, sq, out_white, out_black);
            collect_defender_features_at_both(board, wksq, bksq, sq, touched_mask, out_white, out_black);

            slider_candidates |= MoveGen::generate_rook_moves(sq, occ) & rooks_queens;
            slider_candidates |= MoveGen::generate_bishop_moves(sq, occ) & bishops_queens;
        }
        slider_candidates &= ~touched_mask;

        while (slider_candidates)
        {
            const int sq = cpu::pop_lsb(slider_candidates);
            collect_attacker_features_from_both(board, wksq, bksq, sq, out_white, out_black);
        }
    }

    // Full scoped-recompute entry point for one board snapshot (either strictly
    // before or strictly after a non-king move): `touched_squares` are the
    // squares whose occupant changed (from/to/en-passant-capture square).
    //
    // Instead of rescanning every slider on the board (O(sliders in play) per
    // move), we only need to recompute the outgoing threats of sliders whose
    // ray could actually be affected by this move: a slider's attack set can
    // only change if one of the touched squares is its nearest blocker along
    // some ray -- i.e. exactly the sliders that currently "see" a touched
    // square T under this board's real occupancy. For each touched square we
    // find those candidates in O(1) via the same magic-bitboard/pext attack
    // generator used elsewhere (rook_attacks(T, occ) & (rooks|queens),
    // bishop_attacks(T, occ) & (bishops|queens)); any slider not in this set
    // for T has some other blocker strictly between it and T, so T's
    // occupancy change cannot affect what that slider currently attacks.
    template <Color perspective>
    inline void collect_move_scoped_features(const Board &board, const FixedIntList<MAX_TOUCHED_SQUARES> &touched_squares, FixedIntList<MAX_THREAT_FEATURES> &out)
    {
        const int ksq = board.king_sq[perspective];
        const U64 occ = board.occupancies[NO_COLOR];
        const U64 rooks_queens = board.pieces_occ[get_piece_index(ROOK, WHITE)] | board.pieces_occ[get_piece_index(ROOK, BLACK)] |
                                 board.pieces_occ[get_piece_index(QUEEN, WHITE)] | board.pieces_occ[get_piece_index(QUEEN, BLACK)];
        const U64 bishops_queens = board.pieces_occ[get_piece_index(BISHOP, WHITE)] | board.pieces_occ[get_piece_index(BISHOP, BLACK)] |
                                   board.pieces_occ[get_piece_index(QUEEN, WHITE)] | board.pieces_occ[get_piece_index(QUEEN, BLACK)];

        // See collect_move_scoped_features_both's touched_mask note: a touched
        // square that is itself a slider is already covered by the loop
        // above, so it's masked out of slider_candidates before the final
        // pass to avoid re-emitting its features a second time.
        // touched_mask is computed upfront -- see collect_move_scoped_features_both's
        // note on why the full set must be known before any defender scan.
        U64 touched_mask = 0ULL;
        for (int sq : touched_squares)
            touched_mask |= (1ULL << sq);

        U64 slider_candidates = 0ULL;
        for (int sq : touched_squares)
        {
            collect_attacker_features_from<perspective>(board, ksq, sq, out);
            collect_defender_features_at<perspective>(board, ksq, sq, touched_mask, out);

            slider_candidates |= MoveGen::generate_rook_moves(sq, occ) & rooks_queens;
            slider_candidates |= MoveGen::generate_bishop_moves(sq, occ) & bishops_queens;
        }
        slider_candidates &= ~touched_mask;

        while (slider_candidates)
        {
            const int sq = cpu::pop_lsb(slider_candidates);
            collect_attacker_features_from<perspective>(board, ksq, sq, out);
        }
    }

}
