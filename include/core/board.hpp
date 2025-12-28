#pragma once

#include <stdexcept>

#include "engine/zobrist.hpp"

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
    uint16_t halfmove_clock;
    int last_irreversible_move_index;
    Move move;
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
    uint8_t castling_rights;     // Castle rights
    uint8_t en_passant_sq;       // En passant capture case (255 = none)
    uint16_t halfmove_clock = 0; // For 50 moves rule
    uint32_t fullmove_number;
    Color side_to_move; // White or black turn ?

    std::vector<UndoInfo> history;

    U64 zobrist_key;

    int last_irreversible_move_index = 0;

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
        zobrist_key ^= zobrist_black_to_move;
        side_to_move = static_cast<Color>(1 - static_cast<int>(side_to_move));
    }

    inline bitboard &get_piece_bitboard(const Color color, const Piece type)
    {
        assert(type <= KING);
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
    bool play(const Move &move);
    char piece_to_char(Color color, Piece type) const;
    void show() const;

    /// @brief Checks if specified square is being occupied
    /// @param sq
    /// @param piece (NO_PIECE == checks for any piece type)
    /// @param color (NO_COLOR == checks for any color)
    /// @return true if occupied
    inline bool is_occupied(int sq, Piece piece, Color color) const
    {
        const U64 mask = sq_mask(sq);

        if (piece == NO_PIECE)
        {
            if (color == WHITE)
                return occupied_white & mask;
            if (color == BLACK)
                return occupied_black & mask;
            return occupied_all & mask; // NO_COLOR
        }

        if (color == WHITE)
            return get_piece_bitboard(WHITE, piece) & mask;
        if (color == BLACK)
            return get_piece_bitboard(BLACK, piece) & mask;

        // NO_COLOR: either side
        return (get_piece_bitboard(WHITE, piece) |
                get_piece_bitboard(BLACK, piece)) &
               mask;
    }

    bool is_occupied(const int sq, const int piece, const Color color) const
    {
        return is_occupied(sq, static_cast<Piece>(piece), color);
    }

    void unplay(const Move move);

    bool is_repetition() const;

    inline uint8_t get_castling_rights() const
    {
        return castling_rights;
    }
    inline uint8_t get_en_passant_sq() const
    {
        return en_passant_sq;
    }

    inline U64 get_hash() const
    {
        return zobrist_key;
    }

    void compute_full_hash();

    inline void play_null_move(int &stored_ep_sq)
    {
        if (en_passant_sq != EN_PASSANT_SQ_NONE)
            zobrist_key ^= zobrist_en_passant[en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
        stored_ep_sq = en_passant_sq;
        en_passant_sq = EN_PASSANT_SQ_NONE;
        zobrist_key ^= zobrist_en_passant[8];

        switch_trait();
    }
    inline void unplay_null_move(int stored_ep_sq)
    {
        switch_trait();
        zobrist_key ^= zobrist_en_passant[8];
        en_passant_sq = stored_ep_sq;
        if (en_passant_sq != EN_PASSANT_SQ_NONE)
            zobrist_key ^= zobrist_en_passant[en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
    }

    void undo_last_move();

    uint16_t get_halfmove_clock()
    {
        return halfmove_clock;
    }

    inline int get_history_size()
    {
        return history.size();
    }

    int get_smallest_attacker(U64 all_attackers, Color side, Piece &found_type)
    {
        U64 side_attackers = all_attackers & (side == WHITE ? occupied_white : occupied_black);
        if (!side_attackers)
            return -1;

        // Ordre de priorité : Pion < Cavalier < Fou < Tour < Dame < Roi
        const int offset = side == WHITE ? 0 : N_PIECES_TYPE_HALF;
        for (int type = PAWN; type <= KING; ++type)
        {
            U64 subset = side_attackers & pieces_occ[type + offset];
            if (subset)
            {
                found_type = static_cast<Piece>(type);
                return get_lsb_index(subset);
            }
        }
        return -1;
    }
};