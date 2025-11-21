#pragma once

#include "piece.hpp"

class Move
{
private:
    // All move data is stored in a single 32-bit integer for maximum efficiency
    uint32_t value;

public:
    // --- Flags for Move Encoding (Example: 4 bits for type/special action) ---
    // These are bit masks used to extract or set specific information within the 32-bit 'value'
    enum Flags : uint32_t
    {
        // Simple move types
        NONE = 0x00,
        QUIET_MOVE = 0x00,

        // Special moves flags (must fit within the 4 flag bits)
        DOUBLE_PUSH = 0x01, // Pawn moves two steps (potential En Passant)
        KING_CASTLE = 0x02,
        QUEEN_CASTLE = 0x03,
        CAPTURE = 0x04,
        EN_PASSANT_CAP = 0x05,

        // Promotion flag (Promotion and capture + promotion are differentiated later)
        PROMOTION_MASK = 0x80, // Example mask for promotions

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
        // Add encoding for captured piece and promotion piece if needed
    }
    explicit Move(Square from_sq, Square to_sq, Piece from_piece, Flags flags = NONE):
        Move(static_cast<int>(from_sq), static_cast<int>(to_sq), from_piece, flags){}

    // --- Accessors (Read-only) ---

    // Gets the starting square (index 0-63)
    inline int get_from_sq() const
    {
        return value & 0x3F; // Mask for bits 0-5
    }

    // Gets the destination square (index 0-63)
    inline int get_to_sq() const
    {
        return (value >> 6) & 0x3F; // Shift 6 bits and mask
    }

    inline Piece get_from_piece() const
    {
        return static_cast<Piece>((value >> 16) & 0xF);
    }

    // Checks if the move is a castling move
    inline bool is_castling() const
    {
        uint32_t flag = (value >> 12) & 0xF; // Extract the 4 flag bits
        return (flag == KING_CASTLE || flag == QUEEN_CASTLE);
    }

    // Checks if the move is a promotion
    inline bool is_promotion() const
    {
        return (value & PROMOTION_MASK);
    }

    // Returns the entire 32-bit encoded move value
    inline uint32_t get_value() const
    {
        return value;
    }
};