#pragma once

#include <string>
#include <stdexcept>

#include "core/piece/piece.hpp"

class Move
{
private:
    uint32_t value;

public:
    enum class MoveError
    {
        InvalidFormat,
        IllegalMove,
        SquareOutOfBounds
    };

    enum Flags : uint32_t
    {
        NONE = 0x00,
        QUIET_MOVE = 0x00,

        // Special moves flags (must fit within the 4 flag bits)
        DOUBLE_PUSH = 0x01, // Pawn moves two steps (potential En Passant)
        KING_CASTLE = 0x02,
        QUEEN_CASTLE = 0x03,
        EN_PASSANT_CAP = 0x04,
        CAPTURE = 0x05,

        // Promotion flag (Promotion and capture + promotion are differentiated later)
        PROMOTION_MASK = 0x06,
    };

    // Default constructor (creates an invalid/null move)
    Move() = default;

    // Constructor to encode a move from its components
    Move(int from_sq, int to_sq, Piece from_piece, Flags flags = NONE, Piece to_piece = NO_PIECE, Piece prom_piece = QUEEN)
    {
        // [0..5] : From Square (6 bits)
        // [6..11]: To Square (6 bits)
        // [12..15]: Flags / Move Type (4 bits)
        // [16..19]: From Piece (4 bits)
        // [20..22]: To Piece (3 bits)
        // [23..26]: Promotion piece

        value = (uint32_t)(from_sq) |
                ((uint32_t)(to_sq) << 6) |
                ((uint32_t)(flags) << 12) |
                ((uint32_t)(from_piece) << 16) |
                ((uint32_t)(to_piece) << 20) |
                ((uint32_t)(prom_piece) << 23);
    }

    Move(uint32_t raw_value) : value(raw_value) {}

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
        const uint32_t flag = get_flags();
        return (flag == KING_CASTLE || flag == QUEEN_CASTLE);
    }

    inline uint32_t get_flags() const
    {
        const uint32_t flag = (value >> 12) & 0xF;
        return flag;
    }

    inline Piece get_to_piece() const
    {
        const Piece to_piece = static_cast<Piece>((value >> 20) & 0x7);
        return to_piece;
    }

    inline bool is_promotion() const
    {
        const uint32_t flag = (value >> 12) & 0xF;
        return (flag == PROMOTION_MASK);
    }

    inline uint32_t get_value() const
    {
        return value;
    }

    inline void set_flags(uint32_t flags)
    {
        const uint32_t mask = ~(0xF << 12);
        value = (mask & value) | (flags << 12);
    }

    inline void set_to_piece(Piece piece)
    {
        const uint32_t mask = ~(0x7 << 20);
        value = (mask & value) | (static_cast<uint32_t>(piece) << 20);
    }
    inline bool has_flag(const Flags flags) const
    {
        return flags == get_flags();
    }

    inline bool is_capture() const
    {
        return get_to_piece() != NO_PIECE;
    }

    std::string to_uci() const
    {
        if (value == 0)
            return "0000"; // Cas d'un coup nul (null move)

        std::string uci = "";

        int from = get_from_sq();
        int to = get_to_sq();

        // Conversion des coordonnées en caractères (file: a-h, rank: 1-8)
        uci += (char)('a' + (from % 8));
        uci += (char)('1' + (from / 8));
        uci += (char)('a' + (to % 8));
        uci += (char)('1' + (to / 8));

        // Gestion de la promotion (UCI exige d'indiquer la pièce de promotion en minuscule)
        if (is_promotion())
        {
            switch (get_promo_piece())
            {
            case QUEEN:
                uci += 'q';
                break;
            case KNIGHT:
                uci += 'n';
                break;
            case BISHOP:
                uci += 'b';
                break;
            case ROOK:
                uci += 'r';
                break;
            default:
                throw std::logic_error("Unsupported promotion piece type");
            }
        }

        return uci;
    }

    Piece get_promo_piece() const
    {
        const Piece piece = static_cast<Piece>((value >> 23) & 0xF);
        return piece;
    }
};

inline bool operator==(const Move &a, const Move &b) noexcept
{
    return a.get_value() == b.get_value();
}
