#pragma once

#include <stdexcept>

#include "core/move/history.hpp"

enum CastlingRights : uint8_t
{
    WHITE_KINGSIDE = 0b0001,  // 1
    WHITE_QUEENSIDE = 0b0010, // 2
    BLACK_KINGSIDE = 0b0100,  // 4
    BLACK_QUEENSIDE = 0b1000, // 8
    ALL_CASTLING = 0b1111     // 15
};

class Board
{
private:
    static constexpr uint8_t PIECE_MASK = 0x07; // 0b00000111
    static constexpr uint8_t COLOR_SHIFT = 3;   // 0b00001000
    static constexpr uint8_t COLOR_MASK = 3;
    static constexpr uint8_t EMPTY_SQ = (NO_COLOR << COLOR_SHIFT) | NO_PIECE;

public:
    // Bitboards for each pieces

    // Bitboards for fast occupancy queries
    bitboard occupancies[3]; // [0] WHITE, [1] BLACK, [2] ALL (NO_COLOR)
    U64 zobrist_key;

    struct State
    {
        uint8_t castling_rights;
        uint8_t en_passant_sq;
        uint16_t halfmove_clock;
        Color side_to_move;
        int last_irreversible_index;
    } state;

    std::array<bitboard, N_PIECES_TYPE> pieces_occ;

    EvalState eval_state;

    uint8_t mailbox[64];
    History *history_tagged;

    inline Piece get_p(int sq) const { return static_cast<Piece>(mailbox[sq] & PIECE_MASK); }
    inline Color get_c(int sq) const { return static_cast<Color>((mailbox[sq] >> COLOR_SHIFT) & COLOR_MASK); }

    inline History *get_history() const
    {
        return reinterpret_cast<History *>(
            reinterpret_cast<uintptr_t>(history_tagged) & ~1ULL);
    }

    inline bool owns_history() const
    {
        return reinterpret_cast<uintptr_t>(history_tagged) & 1ULL;
    }
    // Rule of five
    Board()
    {
        History *raw = new History();
        history_tagged = reinterpret_cast<History *>(
            reinterpret_cast<uintptr_t>(raw) | 1ULL);
    }

    // 1. Copy
    Board(const Board &other)
    {
        std::memcpy(occupancies, other.occupancies, sizeof(occupancies));
        std::memcpy(&state, &other.state, sizeof(state));
        pieces_occ = other.pieces_occ;
        eval_state = other.eval_state;
        std::memcpy(mailbox, other.mailbox, sizeof(mailbox));
        zobrist_key = other.zobrist_key;

        History *raw_copy = new History(*other.get_history());
        history_tagged = reinterpret_cast<History *>(
            reinterpret_cast<uintptr_t>(raw_copy) | 1ULL);
    }
    // 2. Copy affectation operator
    Board &operator=(const Board &other)
    {
        if (this != &other)
        {
            if (history_tagged && owns_history())
                delete get_history();
            History *raw_copy = new History(*other.get_history());
            std::memcpy(occupancies, other.occupancies, sizeof(occupancies));
            std::memcpy(&state, &other.state, sizeof(state));
            pieces_occ = other.pieces_occ;
            eval_state = other.eval_state;
            zobrist_key = other.zobrist_key;
            std::memcpy(mailbox, other.mailbox, sizeof(mailbox));
            history_tagged = reinterpret_cast<History *>(
                reinterpret_cast<uintptr_t>(raw_copy) | 1ULL);
        };

        return *this;
    }

    // 3 - Destructor
    ~Board()
    {
        if (history_tagged && owns_history())
        {
            delete get_history();
        }
    }

    // 4. Move
    Board(Board &&other) noexcept : history_tagged(other.history_tagged)
    {
        other.history_tagged = nullptr;
    }
    // 5. Affectation & move operator
    Board &operator=(Board &&other) noexcept
    {
        if (this != &other)
        {
            if (history_tagged && owns_history())
                delete get_history();
            history_tagged = other.history_tagged;
            other.history_tagged = nullptr;
        }
        return *this;
    }
    bool load_fen(const std::string_view fen_string);

    void attach_history(History *h)
    {
        if (history_tagged && owns_history())
        {
            delete get_history();
        }
        history_tagged = h;
    }

    void clear();

    inline void switch_trait()
    {
        zobrist_key ^= zobrist_black_to_move;
        state.side_to_move = static_cast<Color>(1 - static_cast<int>(state.side_to_move));
    }

    inline bitboard &get_piece_bitboard(const Color color, const Piece type)
    {
        assert(type <= KING);
        size_t zero_based_index = (color * N_PIECES_TYPE_HALF) + (type);

        return pieces_occ[zero_based_index];
    }

    inline Color get_side_to_move() const
    {
        return state.side_to_move;
    }

    inline uint8_t get_castling_rights()
    {
        return state.castling_rights;
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
        return occupancies[c];
    }
    inline const bitboard &get_occupancy(Color c) const
    {
        return occupancies[c];
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
        occupancies[WHITE] = EMPTY_MASK;
        occupancies[BLACK] = EMPTY_MASK;

        for (int i{PAWN}; i <= KING; ++i)
        {
            occupancies[WHITE] |= get_piece_bitboard(WHITE, static_cast<Piece>(i));
            occupancies[BLACK] |= get_piece_bitboard(BLACK, static_cast<Piece>(i));
        }

        // Total
        occupancies[NO_COLOR] = occupancies[WHITE] | occupancies[BLACK];
    }
    inline std::pair<Color, Piece> get_piece_on_square(int sq) const
    {
        uint8_t val = mailbox[sq];
        if (val == EMPTY_SQ)
            return {NO_COLOR, NO_PIECE};
        return {static_cast<Color>((val >> COLOR_SHIFT) & COLOR_MASK), static_cast<Piece>(val & PIECE_MASK)};
    }

    inline bool is_occupied(int sq, Piece piece, Color color) const
    {
        uint8_t val = mailbox[sq];
        if (piece == NO_PIECE) [[likely]]
            return val != EMPTY_SQ && (color == NO_COLOR || (val >> COLOR_SHIFT) == color);

        uint8_t target = (color << COLOR_SHIFT) | piece;
        return val == target;
    }

    bool play(const Move move);
    bool is_move_legal(const Move move);
    char piece_to_char(Color color, Piece type) const;
    void show() const;

    bool is_occupied(const int sq, const int piece, const Color color) const
    {
        return is_occupied(sq, static_cast<Piece>(piece), color);
    }

    void unplay(const Move move);

    bool is_repetition() const;

    inline uint8_t get_castling_rights() const
    {
        return state.castling_rights;
    }
    inline uint8_t get_en_passant_sq() const
    {
        return state.en_passant_sq;
    }

    inline U64 get_hash() const
    {
        return zobrist_key;
    }

    void compute_full_hash();

    inline void play_null_move(int &stored_ep_sq)
    {
        if (state.en_passant_sq != EN_PASSANT_SQ_NONE)
            zobrist_key ^= zobrist_en_passant[state.en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
        stored_ep_sq = state.en_passant_sq;
        state.en_passant_sq = EN_PASSANT_SQ_NONE;
        zobrist_key ^= zobrist_en_passant[8];

        switch_trait();
    }
    inline void unplay_null_move(int stored_ep_sq)
    {
        switch_trait();
        zobrist_key ^= zobrist_en_passant[8];
        state.en_passant_sq = stored_ep_sq;
        if (state.en_passant_sq != EN_PASSANT_SQ_NONE)
            zobrist_key ^= zobrist_en_passant[state.en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
    }

    void undo_last_move();

    uint16_t get_halfmove_clock()
    {
        return state.halfmove_clock;
    }

    inline int get_history_size()
    {
        return get_history()->size();
    }

    int get_smallest_attacker(U64 all_attackers, Color side, Piece &found_type)
    {
        U64 side_attackers = all_attackers & occupancies[side];
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

    inline const EvalState &get_eval_state() const
    {
        return eval_state;
    }

    inline const std::array<bitboard, N_PIECES_TYPE> get_all_bitboards() const
    {
        return pieces_occ;
    }

    bool is_king_attacked(Color c);
    bool is_attacked(int sq, Color attacker) const;

    inline U64 get_hash_after(Move m) const
    {
        U64 h = zobrist_key;

        // 1. Inversion du trait (OBLIGATOIRE)
        // On XOR la clé qui représente le tour de jouer
        h ^= zobrist_black_to_move;

        // 2. Déplacement de la pièce
        h ^= zobrist_table[state.side_to_move * N_PIECES_TYPE_HALF + m.get_from_piece()][m.get_from_sq()];
        h ^= zobrist_table[state.side_to_move * N_PIECES_TYPE_HALF + m.get_from_piece()][m.get_to_sq()];

        // 3. Gestion de la capture
        if (m.get_to_piece() != NO_PIECE)
            h ^= zobrist_table[!state.side_to_move * N_PIECES_TYPE_HALF + m.get_to_piece()][m.get_to_sq()];

        return h;
    }

    void verify_consistency()
    {
        U64 expected_all = occupancies[WHITE] | occupancies[BLACK];
        if (occupancies[NO_COLOR] != expected_all)
        {
            throw std::logic_error("Corrupted bitboards");
        }
    }
};