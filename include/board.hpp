#pragma once

#include <stdexcept>

#include "move.hpp"

#define N_PIECES_TYPE 12
#define N_PIECES_TYPE_HALF 6

enum CastlingRights : uint8_t
{
    WHITE_KINGSIDE = 0b0001,  // 1
    WHITE_QUEENSIDE = 0b0010, // 2
    BLACK_KINGSIDE = 0b0100,  // 4
    BLACK_QUEENSIDE = 0b1000, // 8
    ALL_CASTLING = 0b1111     // 15
};

struct UndoInfo
{
    U64 zobrist_key;
    uint8_t castling_rights;
    uint8_t en_passant_sq;
    uint16_t halfmove_clock;
    // We specify which piece has been captured
    Piece captured_piece;
    Color side_to_move;
    Move move; // Full move number (uint32_t)
};

class Board
{
private:
    // Bitboards for each pieces
    std::array<bitboard, N_PIECES_TYPE> pieces_occ;

    // Bitboards for fast occupancy queries
    bitboard occupied_white;
    bitboard occupied_black;
    bitboard occupied_all;

    // State info
    uint8_t castling_rights; // Castle rights
    uint8_t en_passant_sq;   // En passant capture case (0 = none)
    uint16_t halfmove_clock; // For 50 moves rule
    uint32_t fullmove_number;
    Color side_to_move; // White or black turn ?

    std::vector<UndoInfo> move_stack;

    U64 zobrist_key;

public:
    // Rule of five
    Board() = default; // Default constructor. We will use load_fen

    // 1. Copy
    Board(const Board &other) = default;
    // 2. Copy affectation operator
    Board &operator=(const Board &other) = default;

    // 3. Destructor
    ~Board() = default;

    // 4. Move
    Board(Board &&other) noexcept = default;
    // 5. Affectation & move operator
    Board &operator=(Board &&other) noexcept = default;

    bool load_fen(const std::string_view fen_string);

    void clear();

    inline void switch_trait()
    {
        side_to_move = static_cast<Color>(1 - static_cast<int>(side_to_move));
    }

    inline bitboard &get_piece_bitboard(const Color color, const Piece type)
    {
        if (type > KING) // Should be disabled on prod for increased performances
        {
            throw std::out_of_range("Invalid piece type requested for Bitboard access.");
        }
        size_t zero_based_index = (color * N_PIECES_TYPE_HALF) + (type);

        return pieces_occ[zero_based_index];
    }

    inline Color get_side_to_move() const
    {
        return side_to_move;
    }

    inline uint8_t get_castling_rights()
    {
        return castling_rights;
    }

    inline bitboard &get_piece_bitboard(const Color color, const int type)
    {
        return get_piece_bitboard(color, static_cast<Piece>(type));
    }
    inline const bitboard &get_piece_bitboard(const Color color, const Piece type) const
    {
        if (type > KING) // Should be disabled on prod for increased performances
        {
            throw std::out_of_range("Invalid piece type requested for Bitboard access.");
        }
        size_t zero_based_index = (color * N_PIECES_TYPE_HALF) + (type);

        return pieces_occ[zero_based_index];
    }
    inline const bitboard &get_piece_bitboard(const Color color, const int type) const
    {
        return get_piece_bitboard(color, static_cast<Piece>(type));
    }

    inline bitboard &get_occupancy(Color c)
    {
        switch (c)
        {
        case NO_COLOR:
            return occupied_all;
        case WHITE:
            return occupied_white;
        case BLACK:
            return occupied_black;
        }
        throw std::logic_error("Color unsupported");
    }
    inline const bitboard &get_occupancy(Color c) const
    {
        switch (c)
        {
        case NO_COLOR:
            return occupied_all;
        case WHITE:
            return occupied_white;
        case BLACK:
            return occupied_black;
        }
        throw std::logic_error("Color unsupported");
    }

    inline void update_square_bitboard(Color color, Piece type, int square, bool fill)
    {
        bitboard &bitboard_ref = get_piece_bitboard(color, type);
        if (fill)
            bitboard_ref |= (1ULL << square);
        else
            bitboard_ref &= ~(1ULL << square);
    }
    inline void update_occupancy()
    {
        // Reset
        occupied_white = EMPTY_MASK;
        occupied_black = EMPTY_MASK;

        for (int i{PAWN}; i <= KING; ++i)
        {
            occupied_white |= get_piece_bitboard(WHITE, static_cast<Piece>(i));
            occupied_black |= get_piece_bitboard(BLACK, static_cast<Piece>(i));
        }

        // Total
        occupied_all = occupied_white | occupied_black;
    }
    std::pair<Color, Piece> get_piece_on_square(int sq) const;
    bool play(Move &move);
    char piece_to_char(Color color, Piece type) const;
    void show() const;

    /// @brief Checks if specified square is being occupied
    /// @param sq
    /// @param piece (NO_PIECE == checks for any piece type)
    /// @param color (NO_COLOR == checks for any color)
    /// @return true if occupied
    bool is_occupied(const int sq, const Piece piece, const Color color) const
    {
        if (piece == NO_PIECE)
        {
            if (color == BLACK)
            {
                return is_set(occupied_black, sq);
            }
            if (color == WHITE)
            {
                return is_set(occupied_white, sq);
            }
            return is_set(occupied_all, sq);
        }

        const U64 mask = sq_mask(sq);
        U64 final_mask = EMPTY_MASK;
        if (color == WHITE || color == NO_COLOR)
        {
            final_mask |= get_piece_bitboard(WHITE, piece) & mask;
        }
        if (color == BLACK || color == NO_COLOR)
        {
            final_mask |= get_piece_bitboard(BLACK, piece) & mask;
        }
        return final_mask;
    }
    bool is_occupied(const int sq, const int piece, const Color color) const
    {
        return is_occupied(sq, static_cast<Piece>(piece), color);
    }

    void unplay(Move move);

    inline uint8_t get_castling_rights() const
    {
        return castling_rights;
    }
};