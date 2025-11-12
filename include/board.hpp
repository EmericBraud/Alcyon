#pragma once

#include <cstdint>
#include <array>
#include <string>

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

class Board
{
private:
    // Bitboards for each pieces
    std::array<U64, 12> piece_bitboards;

    // Bitboards for fast occupancy queries
    U64 occupied_white;
    U64 occupied_black;
    U64 occupied_all;

    // State info
    uint8_t castling_rights;  // Castle rights
    uint8_t en_passant_sq;    // En passant capture case (0 = none)
    uint16_t halfmove_clock;  // For 50 moves rule
    uint32_t fullmove_number; // Full move number
    Color side_to_move;       // White or black turn ?

    U64 zobrist_key;

    struct UndoInfo
    {
        U64 zobrist_key;
        uint8_t castling_rights;
        uint8_t en_passant_sq;
        uint16_t halfmove_clock;
        // We specify which piece has been captured
        Piece captured_piece;
        Color side_to_move;
    };

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
    inline U64 &get_piece_bitboard(Color color, Piece type);
    inline void update_square_bitboard(Color color, Piece type, int square, bool fill);
    inline void update_occupancy();
    std::pair<Color, Piece> get_piece_on_square(int sq) const;
    char piece_to_char(Color color, Piece type) const;
    void show() const;
};