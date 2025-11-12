#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <stdexcept>
#include <vector>

#include "move.hpp"

#define N_PIECES_TYPE 12
#define N_PIECES_TYPE_HALF 6

using U64 = std::uint64_t;

enum CastlingRights : uint8_t
{
    WHITE_KINGSIDE = 0b0001,  // 1
    WHITE_QUEENSIDE = 0b0010, // 2
    BLACK_KINGSIDE = 0b0100,  // 4
    BLACK_QUEENSIDE = 0b1000, // 8
    ALL_CASTLING = 0b1111     // 15
};

enum Piece : uint8_t
{
    NO_PIECE,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
};

enum Color : uint8_t
{
    WHITE,
    BLACK,
    NO_COLOR
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
    std::array<U64, N_PIECES_TYPE> piece_bitboards;

    // Bitboards for fast occupancy queries
    U64 occupied_white;
    U64 occupied_black;
    U64 occupied_all;

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
    inline U64 &get_piece_bitboard(Color color, Piece type)
    {
        if (type <= NO_PIECE || type > KING) // Should be disabled on prod for increased performances
        {
            throw std::out_of_range("Invalid piece type requested for Bitboard access.");
        }
        size_t zero_based_index = (color * N_PIECES_TYPE_HALF) + (type - 1);

        return piece_bitboards[zero_based_index];
    }

    inline void update_square_bitboard(Color color, Piece type, int square, bool fill)
    {
        U64 &bitboard_ref = get_piece_bitboard(color, type);
        if (fill)
            bitboard_ref |= (1ULL << square);
        else
            bitboard_ref &= ~(1ULL << square);
    }
    inline void update_occupancy()
    {
        // Reset
        occupied_white = 0ULL;
        occupied_black = 0ULL;

        // White 0-5
        for (int i = 0; i < N_PIECES_TYPE_HALF; ++i)
        {
            occupied_white |= piece_bitboards[i];
        }

        // Black (6-12)
        for (int i = N_PIECES_TYPE_HALF; i < N_PIECES_TYPE; ++i)
        {
            occupied_black |= piece_bitboards[i];
        }

        // Total
        occupied_all = occupied_white | occupied_black;
    }
    std::pair<Color, Piece> get_piece_on_square(int sq) const;
    char piece_to_char(Color color, Piece type) const;
    void show() const;
};