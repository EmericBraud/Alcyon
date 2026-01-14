#pragma once

#include "core/board.hpp"

namespace MoveGen
{
    alignas(64) extern std::array<U64, BOARD_SIZE> KnightAttacks;
    alignas(64) extern std::array<U64, BOARD_SIZE> KingAttacks;
    alignas(64) extern std::array<U64, BOARD_SIZE> RookMasks;
    alignas(64) extern std::array<U64, BOARD_SIZE> BishopMasks;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnAttacksWhite;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnAttacksBlack;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnPushWhite;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnPushBlack;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnPush2White;
    alignas(64) extern std::array<U64, BOARD_SIZE> PawnPush2Black;

#ifdef __BMI2__
    struct MagicPEXT
    {
        U64 mask;
        long unsigned int index_start;
    };
    extern std::array<MagicPEXT, BOARD_SIZE> RookMagics;
    extern std::array<MagicPEXT, BOARD_SIZE> BishopMagics;

#else
    struct Magic
    {
        U64 mask;
        U64 magic;
        int shift;
        long unsigned int index_start;
    };

    extern std::array<Magic, BOARD_SIZE> RookMagics;
    extern std::array<Magic, BOARD_SIZE> BishopMagics;

#endif

    extern std::array<U64, ROOK_ATTACKS_SIZE> RookAttacks;
    extern std::array<U64, BISHOP_ATTACKS_SIZE> BishopAttacks;

    void initialize_bitboard_tables();

    void generate_pawn_moves(Board &board, const Color color, MoveList &list);
    inline U64 generate_rook_moves(int from_sq, const bitboard occupancy)
    {
#ifdef __BMI2__
        const MoveGen::MagicPEXT magic = MoveGen::RookMagics[from_sq];

        int index = (int)_pext_u64(occupancy, magic.mask);
        return RookAttacks[magic.index_start + index];

#else
        const MoveGen::Magic magic = MoveGen::RookMagics[from_sq];

        const U64 index = (((occupancy & magic.mask) * magic.magic) >> magic.shift);

        return MoveGen::RookAttacks[index + magic.index_start];
#endif
    }

    inline U64 generate_bishop_moves(int from_sq, const bitboard occupancy)
    {
#ifdef __BMI2__
        const MoveGen::MagicPEXT magic = MoveGen::BishopMagics[from_sq];

        int index = (int)_pext_u64(occupancy, magic.mask);
        return BishopAttacks[magic.index_start + index];
#else
        const MoveGen::Magic magic = MoveGen::BishopMagics[from_sq];

        const U64 index = (((occupancy & magic.mask) * magic.magic) >> magic.shift);

        return MoveGen::BishopAttacks[index + magic.index_start];
#endif
    }

    template <Piece piece_type>
    inline U64 get_pseudo_moves_mask(const Board &board, const int sq, const Color color)
    {
        const U64 occ = board.get_occupancy<NO_COLOR>();
        U64 target_mask = 0;

        if constexpr (piece_type == PAWN)
        {
            const int ep_sq = board.get_en_passant_sq();
            if (color == WHITE)
            {
                // Poussée simple : case devant vide
                U64 push1 = (1ULL << (sq + 8)) & ~occ;
                // Poussée double : push1 possible ET case cible vide ET rangée 2
                U64 push2 = (push1 << 8) & ~occ & 0x00000000FF000000ULL;
                target_mask = push1 | push2;
                target_mask |= PawnAttacksWhite[sq] & board.get_occupancy(BLACK);
                if (ep_sq != EN_PASSANT_SQ_NONE)
                    target_mask |= PawnAttacksWhite[sq] & (1ULL << ep_sq);
            }
            else
            {
                U64 push1 = (1ULL << (sq - 8)) & ~occ;
                U64 push2 = (push1 >> 8) & ~occ & 0x000000FF00000000ULL;
                target_mask = push1 | push2;
                target_mask |= PawnAttacksBlack[sq] & board.get_occupancy(WHITE);
                if (ep_sq != EN_PASSANT_SQ_NONE)
                    target_mask |= PawnAttacksBlack[sq] & (1ULL << ep_sq);
            }
        }
        else if constexpr (piece_type == KNIGHT)
        {
            target_mask = KnightAttacks[sq];
        }
        else if constexpr (piece_type == BISHOP)
        {
            target_mask = generate_bishop_moves(sq, occ);
        }
        else if constexpr (piece_type == ROOK)
        {
            target_mask = generate_rook_moves(sq, occ);
        }
        else if constexpr (piece_type == QUEEN)
        {
            target_mask = generate_rook_moves(sq, occ) | generate_bishop_moves(sq, occ);
        }
        else if constexpr (piece_type == KING)
        {
            target_mask = KingAttacks[sq];
        }
        else
        {

            throw std::logic_error("Piece type unsupported");
        }

        return target_mask & ~board.get_occupancy(color);
    }
    inline U64 get_pseudo_moves_mask(const Board &board, const int sq)
    {
        const PieceInfo pair = board.get_piece_on_square(sq);
        const Color color = pair.first;
        const Piece piece_type = pair.second;

        switch (piece_type)
        {
        case PAWN:
            return get_pseudo_moves_mask<PAWN>(board, sq, color);
        case KNIGHT:
            return get_pseudo_moves_mask<KNIGHT>(board, sq, color);
        case BISHOP:
            return get_pseudo_moves_mask<BISHOP>(board, sq, color);
        case ROOK:
            return get_pseudo_moves_mask<ROOK>(board, sq, color);
        case QUEEN:
            return get_pseudo_moves_mask<QUEEN>(board, sq, color);
        case KING:
            return get_pseudo_moves_mask<KING>(board, sq, color);
        default:
            return 0ULL;
        }
    }
    void initialize_rook_masks();
    void initialize_bishop_masks();
    void initialize_pawn_masks();

    template <Color Us>
    void generate_pseudo_legal_moves(Board &board, MoveList &list);
    template <Color Us>
    void generate_castle_moves(Board &board, MoveList &list);
    template <Color Us>
    void generate_pseudo_legal_captures(const Board &board, MoveList &list);
    U64 get_legal_moves_mask(Board &board, int from_sq);

    template <Color Us>
    void generate_legal_moves(Board &board, MoveList &list);

    inline void generate_legal_moves(Board &board, MoveList &list)
    {
        if (board.get_side_to_move() == WHITE)
        {
            generate_legal_moves<WHITE>(board, list);
            return;
        }
        generate_legal_moves<BLACK>(board, list);
    }
#ifdef __BMI2__
    void export_attack_tables();
    void load_attack_tables();

#else
    void export_attack_table(const std::array<MoveGen::Magic, BOARD_SIZE> m_array, bool is_rook);
    void run_magic_searcher();

    /// @brief Gets the piece attack's sizes to then write them as static in the code (unused on prod)
    /// @param is_rook rook / bishop
    void get_sizes(bool is_rook);

    void load_magics(bool is_rook);

    void load_magics();
    void load_attacks_rook();
    void load_attacks_bishop();
    void load_attacks();

#endif

    inline void init_move_flags(const Board &board, Move &move)
    {
        const int from_sq{move.get_from_sq()}, to_sq{move.get_to_sq()};
        const Piece from_piece{move.get_from_piece()};
        const PieceInfo to_piece_info{board.get_piece_on_square(to_sq)};

        move.set_to_piece(to_piece_info.second);
        if (to_piece_info.second != NO_PIECE)
        {
            move.set_flags(Move::Flags::CAPTURE);
        }

        if (from_piece == PAWN)
        {
            if (to_sq / 8 % 7 == 0) // First or last row
            {
                move.set_flags(Move::Flags::PROMOTION_MASK);
                return;
            }
            else if (abs(from_sq - to_sq) == 16)
            {
                move.set_flags(Move::Flags::DOUBLE_PUSH);
                return;
            }
            else if (to_sq == board.get_en_passant_sq())
            {
                move.set_flags(Move::Flags::EN_PASSANT_CAP);
                return;
            }
        }
    }
    inline void push_moves_from_mask(MoveList &list, int from, Piece type, bitboard targets, const Board &board)
    {
        while (targets)
        {
            int to = pop_lsb(targets); //
            Move m{from, to, type};
            init_move_flags(board, m);
            list.push(m);
        }
    }
    U64 attackers_to(int sq, U64 occupancy, const Board &b);
    U64 update_xrays(int sq, U64 occupied, const Board &board);
}

U64 generate_sliding_attack(int sq, U64 occupancy, bool is_rook);
