#pragma once

#include <array>
#include <cstdint>
#include <string>

#include "common/constants.hpp"
#include "common/mask.hpp"
#include "core/board/board.hpp"
#include "core/piece/piece.hpp"

namespace gpu_eval {

// Plain-old-data snapshot of everything the plane encoder (gpu_encoder.hpp)
// needs, captured by value off a Board so it can cross into the GPU-prep
// thread without any reference into the (concurrently mutated) search
// worker's live board. Trivially copyable, fixed size -- safe to store by
// value in the preallocated queue (gpu_queue.hpp), no allocation.
struct GpuPosition {
    // Indexed by get_piece_index(Piece, Color) = piece + 6*color, same
    // layout as Board::pieces_occ -- see src/core/piece/piece.hpp.
    std::array<U64, constants::NumPieceVariants> pieces_occ{};
    U64 zobrist_key = 0;
    Color side_to_move = WHITE;
    std::uint8_t castling_rights = 0;
    std::uint8_t en_passant_sq = constants::EnPassantSqNone;
    std::uint16_t halfmove_clock = 0;

    static GpuPosition from_board(const Board &board) {
        GpuPosition pos;
        pos.pieces_occ = board.get_all_bitboards();
        pos.zobrist_key = board.get_hash();
        pos.side_to_move = board.get_side_to_move();
        pos.castling_rights = board.get_castling_rights();
        pos.en_passant_sq = board.get_en_passant_sq();
        pos.halfmove_clock = board.get_halfmove_clock();
        return pos;
    }

    U64 piece_bb(Color color, Piece piece) const {
        return pieces_occ[get_piece_index(piece, color)];
    }

    U64 occupancy(Color color) const {
        U64 occ = 0;
        for (int p = 0; p < constants::PieceTypeCount; ++p) {
            occ |= pieces_occ[get_piece_index(static_cast<Piece>(p), color)];
        }
        return occ;
    }

    U64 all_occupancy() const { return occupancy(WHITE) | occupancy(BLACK); }

    // Round-trips through Board::load_fen() on the GPU-prep thread so
    // candidate moves can be applied via the engine's own well-tested
    // Board::play()/unplay() rather than re-deriving move-application
    // (castling rights bookkeeping, en passant, etc.) from scratch here.
    // Not hot-path: called once per PV-leaf task, not per node.
    std::string to_fen() const;
};

} // namespace gpu_eval
