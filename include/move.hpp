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
        EN_PASSANT_CAP = 0x04,
        CAPTURE = 0x05,

        // Promotion flag (Promotion and capture + promotion are differentiated later)
        PROMOTION_MASK = 0x06,

        // ... more specific masks for piece promotion type (N, B, R, Q)
    };

    // Default constructor (creates an invalid/null move)
    Move() : value(0) {}

    // Constructor to encode a move from its components
    Move(int from_sq, int to_sq, Piece from_piece, Flags flags = NONE, Piece to_piece = NO_PIECE, int prev_castling_rights = 0, bool prev_en_passant_flag = false, int prev_en_passant_file = 0)
    {
        // Example encoding scheme:
        // [0..5] : From Square (6 bits)
        // [6..11]: To Square (6 bits)
        // [12..15]: Flags / Move Type (4 bits)
        // [16..19]: From Piece (4 bits)
        // [20..22]: To Piece (3 bits)
        // [23..27]: Prev castling rights (4 bits)
        // [28..29]: Prev en passant flag (1 bit)
        // [29..32]: Prev En Passant File (3 bits) (> Usefull if En passant flag set to 1)

        value = (uint32_t)(from_sq) |
                ((uint32_t)(to_sq) << 6) |
                ((uint32_t)(flags) << 12) |
                ((uint32_t)(from_piece) << 16) |
                ((uint32_t)(to_piece) << 20) |
                ((uint32_t)(prev_castling_rights) << 23) |
                ((uint32_t)(prev_en_passant_flag)) << 28 |
                ((uint32_t)(prev_en_passant_file)) << 29;
    }

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

    inline void set_capture(Piece piece)
    {
        set_to_piece(piece);
    }

    inline bool has_flag(const Flags flags) const
    {
        return flags == get_flags();
    }

    inline int get_prev_castling_rights() const
    {
        return (value >> 23) & 0xF;
    }

    inline void set_prev_castling_rights(uint32_t val)
    {
        constexpr uint32_t SHIFT = 23;
        constexpr uint32_t MASK = 0xFu << SHIFT;

        value = (value & ~MASK) | ((val & 0xFu) << SHIFT);
    }

    inline int get_prev_en_passant(const Color move_side) const
    {
        const bool prev_en_passant_flag = (value >> 28) & 0x01;
        if (!prev_en_passant_flag)
        {
            return EN_PASSANT_SQ_NONE;
        }
        const int en_passant_file = (value >> 29) & 0x7;
        if (move_side == WHITE)
        {
            return Square::a6 + en_passant_file; // Previous black double push
        }
        if (move_side == BLACK)
        {
            return Square::a3 + en_passant_file; // Previous white double push
        }
        throw std::logic_error("Move side can't be NONE when getting en passant sq");
    }
    inline void set_prev_en_passant(const int en_passant_sq)
    {
        if (en_passant_sq == EN_PASSANT_SQ_NONE)
        {
            value &= ~(1ULL << 28); // Set en passant flag to 0
            return;
        }
        value |= (1ULL << 28); // Set en passant flag to 1
        uint32_t mask = ~(0x07 << 29);
        value = (mask & value) | (static_cast<uint32_t>(en_passant_sq % 8) << 29);
    }

    inline bool set_en_passant(const int en_passant_sq)
    {
        const Piece from_piece = get_from_piece();
        const int to_sq = get_to_sq();
        if (from_piece == PAWN && to_sq == en_passant_sq)
        {
            set_flags(Move::EN_PASSANT_CAP);
            return true;
        }
        return false;
    }

    inline bool set_promotion(const Color side_to_move)
    {
        const Piece from_piece = get_from_piece();
        const int to_sq = get_to_sq();
        if (!from_piece == PAWN)
            return false;
        if (side_to_move == WHITE && to_sq / 8 == 7)
        {
            set_flags(Move::PROMOTION_MASK);
            return true;
        }
        if (side_to_move == BLACK && to_sq / 8 == 0)
        {
            set_flags(Move::PROMOTION_MASK);
            return true;
        }
        return false;
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
            // Note: Selon ton encodage, tu devras peut-être stocker quelle pièce est choisie.
            // Si ton moteur promeut par défaut en Dame (Queen) :
            uci += 'q';

            /* Si tu gères plusieurs types de promotion, il faudra extraire l'info de tes flags :
            switch (get_promotion_piece()) {
                case KNIGHT: uci += 'n'; break;
                case BISHOP: uci += 'b'; break;
                case ROOK:   uci += 'r'; break;
                default:     uci += 'q'; break;
            }
            */
        }

        return uci;
    }
};