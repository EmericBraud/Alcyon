#pragma once

#include <stdexcept>
#include <expected>
#include <cstdint>
#include <array>
#include <cstring>

#include "common/mask.hpp"
#include "common/constants.hpp"
#include "common/cpu.hpp"
#include "core/move/history.hpp"
#include "core/move/move_list.hpp"
#include "core/board/zobrist.hpp"

enum CastlingRights : std::uint8_t
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
    static constexpr std::uint8_t PIECE_MASK = 0x07; // 0b00000111
    static constexpr std::uint8_t COLOR_SHIFT = 3;   // 0b00001000
    static constexpr std::uint8_t COLOR_MASK = 3;
    static constexpr std::uint8_t EMPTY_SQ = (NO_COLOR << COLOR_SHIFT) | NO_PIECE;

public:
    // Bitboards for each pieces

    // Bitboards for fast occupancy queries
    U64 occupancies[3]; // [0] WHITE, [1] BLACK, [2] ALL (NO_COLOR)
    U64 zobrist_key;

    struct State
    {
        std::uint8_t castling_rights;
        std::uint8_t en_passant_sq;
        uint16_t halfmove_clock;
        Color side_to_move;
        int last_irreversible_index;
    } state;

    std::array<U64, constants::NumPieceVariants> pieces_occ;

    std::uint8_t mailbox[64];
    std::uint8_t king_sq[2];
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
        std::memcpy(mailbox, other.mailbox, sizeof(mailbox));
        zobrist_key = other.zobrist_key;
        std::memcpy(king_sq, other.king_sq, sizeof(king_sq));

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
            std::memcpy(king_sq, other.king_sq, sizeof(king_sq));
            pieces_occ = other.pieces_occ;
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

    static std::expected<Move, Move::MoveError> parse_move_uci(std::string_view uci, const Board &board)
    {
        if (uci.length() < 4 || uci.length() > 5)
            return std::unexpected(Move::MoveError::InvalidFormat);

        int from_f = uci[0] - 'a';
        int from_r = uci[1] - '1';
        int to_f = uci[2] - 'a';
        int to_r = uci[3] - '1';

        if (from_f < 0 || from_f > 7 || from_r < 0 || from_r > 7 ||
            to_f < 0 || to_f > 7 || to_r < 0 || to_r > 7)
            return std::unexpected(Move::MoveError::SquareOutOfBounds);

        int from_sq = from_r * 8 + from_f;
        int to_sq = to_r * 8 + to_f;

        Piece from_piece = board.get_p(from_sq);
        if (from_piece == NO_PIECE)
            return std::unexpected(Move::MoveError::IllegalMove);

        Piece to_piece = board.get_p(to_sq);
        Move::Flags flags = Move::Flags::NONE;
        if (to_piece != NO_PIECE)
            flags = Move::Flags::CAPTURE;

        // --- DETECTION DES COUPS SPECIAUX ---

        // 1. Double poussée de pion
        if (from_piece == PAWN && std::abs(from_r - to_r) == 2)
            flags = Move::Flags::DOUBLE_PUSH;

        // 2. Roque (détecté par le mouvement du Roi sur 2 colonnes)
        if (from_piece == KING && std::abs(from_f - to_f) == 2)
        {
            flags = (to_f > from_f) ? Move::Flags::KING_CASTLE : Move::Flags::QUEEN_CASTLE;
        }

        // 3. Capture En Passant (Pion qui bouge en diagonale sur une case vide)
        if (from_piece == PAWN && from_f != to_f && to_piece == NO_PIECE)
        {
            flags = Move::Flags::EN_PASSANT_CAP;
            to_piece = PAWN; // La pièce capturée est un pion
        }

        // 4. Promotion
        Piece prom_piece = NO_PIECE;
        if (uci.length() == 5)
        {
            flags = Move::Flags::PROMOTION_MASK;
            switch (uci[4])
            {
            case 'q':
                prom_piece = QUEEN;
                break;
            case 'r':
                prom_piece = ROOK;
                break;
            case 'b':
                prom_piece = BISHOP;
                break;
            case 'n':
                prom_piece = KNIGHT;
                break;
            default:
                return std::unexpected(Move::MoveError::InvalidFormat);
            }
        }

        return Move(from_sq, to_sq, from_piece, flags, to_piece, prom_piece);
    }

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
        zobrist_key ^= zobrist_side_to_move;
        state.side_to_move = static_cast<Color>(1 - static_cast<int>(state.side_to_move));
    }

    inline U64 &get_piece_bitboard(const Color color, const Piece type)
    {
        assert(type <= KING);
        size_t zero_based_index = (color * constants::PieceTypeCount) + (type);

        return pieces_occ[zero_based_index];
    }

    inline Color get_side_to_move() const
    {
        return state.side_to_move;
    }

    inline std::uint8_t get_castling_rights()
    {
        return state.castling_rights;
    }

    inline U64 &get_piece_bitboard(const Color color, const int type)
    {
        return get_piece_bitboard(color, static_cast<Piece>(type));
    }
    inline const U64 &get_piece_bitboard(const Color color, const Piece type) const
    {
        if (type > KING) // Should be disabled on prod for increased performances
        {
            throw std::out_of_range("Invalid piece type requested for Bitboard access.");
        }
        size_t zero_based_index = (color * constants::PieceTypeCount) + (type);

        return pieces_occ[zero_based_index];
    }
    inline const U64 &get_piece_bitboard(const Color color, const int type) const
    {
        return get_piece_bitboard(color, static_cast<Piece>(type));
    }

    template <Color Us, Piece p>
    inline U64 &get_piece_bitboard()
    {
        static_assert(p <= KING, "Invalid piece type");
        constexpr size_t index = (Us * constants::PieceTypeCount) + p;
        return pieces_occ[index];
    }

    template <Color Us, Piece p>
    inline U64 get_piece_bitboard() const
    {
        static_assert(p <= KING, "Invalid piece type");
        constexpr size_t index = (Us * constants::PieceTypeCount) + p;
        return pieces_occ[index];
    }

    inline U64 &get_occupancy(Color c)
    {
        return occupancies[c];
    }
    inline const U64 &get_occupancy(Color c) const
    {
        return occupancies[c];
    }

    template <Color Us>
    inline const U64 &get_occupancy() const
    {
        return occupancies[Us];
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
        occupancies[WHITE] = core::mask::EmptyMask;
        occupancies[BLACK] = core::mask::EmptyMask;

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
        std::uint8_t val = mailbox[sq];
        if (val == EMPTY_SQ)
            return {NO_COLOR, NO_PIECE};
        return {static_cast<Color>((val >> COLOR_SHIFT) & COLOR_MASK), static_cast<Piece>(val & PIECE_MASK)};
    }

    inline bool is_occupied(int sq, Piece piece, Color color) const
    {
        std::uint8_t val = mailbox[sq];
        if (piece == NO_PIECE) [[likely]]
            return val != EMPTY_SQ && (color == NO_COLOR || (val >> COLOR_SHIFT) == color);

        std::uint8_t target = (color << COLOR_SHIFT) | piece;
        return val == target;
    }
    template <Color Us>
    void play(const Move move);

    inline void play(const Move move)
    {
        if (state.side_to_move == WHITE)
        {
            return play<WHITE>(move);
        }
        return play<BLACK>(move);
    }
    template <Color Us>
    bool is_move_legal(const Move move);
    bool is_move_legal(const Move move)
    {
        const Color us = get_side_to_move();
        if (us == WHITE)
            return is_move_legal<WHITE>(move);
        return is_move_legal<BLACK>(move);
    };
    void compute_full_hash();
    char piece_to_char(Color color, Piece type) const;
    void show() const;

    bool en_passant_capture_possible() const;

    uint64_t polyglot_key() const;

    bool is_occupied(const int sq, const int piece, const Color color) const
    {
        return is_occupied(sq, static_cast<Piece>(piece), color);
    }

    template <Color Us>
    void unplay(const Move move);

    inline void unplay(const Move move)
    {
        if (state.side_to_move == WHITE)
        {
            unplay<BLACK>(move);
            return;
        }
        unplay<WHITE>(move);
    }

    bool is_repetition() const;

    bool is_move_pseudo_legal(const Move &move) const;

    inline std::uint8_t get_castling_rights() const
    {
        return state.castling_rights;
    }
    inline std::uint8_t get_en_passant_sq() const
    {
        return state.en_passant_sq;
    }

    inline U64 get_hash() const
    {
        return zobrist_key;
    }

    inline void play_null_move(int &stored_ep_sq)
    {
        if (state.en_passant_sq != constants::EnPassantSqNone)
            zobrist_key ^= zobrist_en_passant[state.en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
        stored_ep_sq = state.en_passant_sq;
        state.en_passant_sq = constants::EnPassantSqNone;
        zobrist_key ^= zobrist_en_passant[8];

        switch_trait();
    }
    inline void unplay_null_move(int stored_ep_sq)
    {
        switch_trait();
        zobrist_key ^= zobrist_en_passant[8];
        state.en_passant_sq = stored_ep_sq;
        if (state.en_passant_sq != constants::EnPassantSqNone)
            zobrist_key ^= zobrist_en_passant[state.en_passant_sq % 8];
        else
            zobrist_key ^= zobrist_en_passant[8];
    }

    void undo_last_move();

    uint16_t get_halfmove_clock() const
    {
        return state.halfmove_clock;
    }

    inline int get_history_size()
    {
        return get_history()->size();
    }

    template <Color Side>
    FORCE_INLINE int get_smallest_attacker(U64 all_attackers, Piece &found_type) const
    {
        // 1. Check rapide : Si l'adversaire n'attaque même pas la case, on sort direct.
        // C'est très fréquent, donc ça vaut le coup de tester tôt.
        U64 side_attackers = all_attackers & occupancies[Side];
        if (!side_attackers)
            return -1;

        const int offset = (Side == WHITE) ? 0 : constants::PieceTypeCount;

        for (int type = PAWN; type <= KING; ++type)
        {
            U64 subset = side_attackers & pieces_occ[type + offset];
            if (subset)
            {
                found_type = static_cast<Piece>(type);
                // Ici, subset est > 0, donc get_lsb_index utilisera l'instruction TZCNT pure
                return cpu::get_lsb_index(subset);
            }
        }
        return -1;
    }

    inline const std::array<U64, constants::NumPieceVariants> get_all_bitboards() const
    {
        return pieces_occ;
    }

    inline bool is_king_attacked(Color c)
    {
        return is_attacked(king_sq[c], (Color)!c);
    }
    template <Color Us>
    inline bool is_king_attacked()
    {
        return is_attacked<!Us>(king_sq[Us]);
    }

    template <Color Attacker>
    bool is_attacked(int sq) const;

    inline bool is_attacked(int sq, Color attacker) const
    {
        if (attacker == WHITE)
        {
            return is_attacked<WHITE>(sq);
        }
        return is_attacked<BLACK>(sq);
    };

    inline U64 get_hash_after(Move m) const
    {
        U64 h = zobrist_key;

        // 1. Inversion du trait (OBLIGATOIRE)
        // On XOR la clé qui représente le tour de jouer
        h ^= zobrist_side_to_move;

        // 2. Déplacement de la pièce
        h ^= zobrist_table[state.side_to_move * constants::PieceTypeCount + m.get_from_piece()][m.get_from_sq()];
        h ^= zobrist_table[state.side_to_move * constants::PieceTypeCount + m.get_from_piece()][m.get_to_sq()];

        // 3. Gestion de la capture
        if (m.get_to_piece() != NO_PIECE)
            h ^= zobrist_table[!state.side_to_move * constants::PieceTypeCount + m.get_to_piece()][m.get_to_sq()];

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