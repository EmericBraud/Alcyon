#pragma once

#include "piece.hpp"

class Move
{
private:
    uint32_t value;

public:
    enum Flags : uint32_t
    {
        NONE = 0x00,
        QUIET_MOVE = 0x00,

        // Special moves flags (must fit within the 4 flag bits)
        DOUBLE_PUSH = 0x01, // Pawn moves two steps (potential En Passant)
        KING_CASTLE = 0x02,
        QUEEN_CASTLE = 0x03,
        CAPTURE = 0x04,
        EN_PASSANT_CAP = 0x05,

        // Promotion flag (Promotion and capture + promotion are differentiated later)
        PROMOTION_MASK = 0x80,

        // ... more specific masks for piece promotion type (N, B, R, Q)
    };

    // Default constructor (creates an invalid/null move)
    Move() : value(0) {}

    // Constructor to encode a move from its components
    Move(int from_sq, int to_sq, Piece from_piece, Flags flags = NONE)
    {
        // Example encoding scheme:
        // [0..5] : From Square (6 bits)
        // [6..11]: To Square (6 bits)
        // [12..15]: Flags / Move Type (4 bits)
        // [16..20]: From Piece (4 bits)

        value = (uint32_t)(from_sq) |
                ((uint32_t)(to_sq) << 6) |
                ((uint32_t)(flags) << 12) |
                ((uint32_t)(from_piece) << 16);
    }
    explicit Move(Square from_sq, Square to_sq, Piece from_piece, Flags flags = NONE) : Move(static_cast<int>(from_sq), static_cast<int>(to_sq), from_piece, flags) {}

    inline int get_from_sq() const
    {
        return value & 0x3F; // Mask for bits 0-5
    }

    inline int get_to_sq() const
    {
        return (value >> 6) & 0x3F; // Shift 6 bits and mask
    }

    inline Piece get_from_piece() const
    {
        return static_cast<Piece>((value >> 16) & 0xF);
    }

    inline bool is_castling() const
    {
        uint32_t flag = (value >> 12) & 0xF; // Extract the 4 flag bits
        return (flag == KING_CASTLE || flag == QUEEN_CASTLE);
    }

    inline bool is_promotion() const
    {
        uint32_t flag = (value >> 12) & 0xF;
        return (flag == PROMOTION_MASK);
    }

    inline uint32_t get_value() const
    {
        return value;
    }
};