#include "board.hpp"

#include <iostream>
#include <cassert>
#include <stdexcept>

void Board::clear()
{
    for (auto &b : pieces_occ)
    {
        b = 0;
    }
    occupied_white = 0;
    occupied_black = 0;
    occupied_all = 0;
    castling_rights = 0;
    en_passant_sq = 0;
    halfmove_clock = 0;
    side_to_move = WHITE;
    zobrist_key = 0;
    fullmove_number = 0;
    move_stack.clear();
}

std::pair<Color, Piece> Board::get_piece_on_square(int sq) const
{
    // Check if the square is occupied at all
    if (!is_set(occupied_all, sq))
    {
        return {NO_COLOR, NO_PIECE};
    }

    // Determine color first
    Color piece_color = is_set(occupied_white, sq) ? WHITE : BLACK;

    // Determine piece type
    for (int type = PAWN; type <= KING; ++type)
    {
        // Check if the piece's bitboard is set at the given square
        if (is_occupied(sq, type, piece_color))
        {
            return {piece_color, static_cast<Piece>(type)};
        }
    }

    throw std::logic_error("Incoherent result, square should be occupied");
}

bool Board::play(Move &move)
{
    const int from_sq{move.get_from_sq()}, to_sq{move.get_to_sq()};
    const Piece from_piece{move.get_from_piece()};
    if (from_piece == NO_PIECE)
    {
        std::cerr << "No piece found on square" << from_sq << std::endl;
        return false;
    }
    const Color opponent_color = (side_to_move == WHITE) ? BLACK : WHITE;
    assert(is_occupied(from_sq, from_piece, side_to_move));
    assert(!is_occupied(to_sq, NO_PIECE, side_to_move));

    const U64 to_sq_mask{sq_mask(to_sq)};

    move.set_prev_castling_rights(castling_rights);

    get_piece_bitboard(side_to_move, from_piece) ^= sq_mask(from_sq); // Delete old position
    get_piece_bitboard(side_to_move, from_piece) |= to_sq_mask;       // Fill new position

    // Delete castling rights flags if rook is moved / captured
    const std::array<std::pair<Square, CastlingRights>, 4> castling_rooks_checker = {
        std::make_pair(Square::a1, WHITE_QUEENSIDE),
        std::make_pair(Square::h1, WHITE_KINGSIDE),
        std::make_pair(Square::a8, BLACK_QUEENSIDE),
        std::make_pair(Square::h8, BLACK_KINGSIDE),
    };
    for (const std::pair<Square, CastlingRights> &p : castling_rooks_checker)
    {
        if ((from_sq == p.first || to_sq == p.first) && castling_rights & p.second)
        {
            const uint8_t mask = ~p.second;
            castling_rights &= mask;
        }
    }
    if (move.has_flag(Move::Flags::KING_CASTLE))
    {
        if (side_to_move == WHITE)
        {
            if (!(castling_rights & WHITE_KINGSIDE))
            {
                throw std::logic_error("Castling rights missing");
            }
            U64 mask = (1ULL << 7) | (1ULL << 5);
            bitboard &rook_bit = get_piece_bitboard(WHITE, ROOK);
            rook_bit ^= mask;
            castling_rights ^= WHITE_KINGSIDE;
            if (castling_rights & WHITE_QUEENSIDE)
            {
                castling_rights ^= WHITE_QUEENSIDE;
            }
            goto play_normal_exit;
        }
        else
        {
            if (!(castling_rights & BLACK_KINGSIDE))
            {
                throw std::logic_error("Castling rights missing");
            }
            U64 mask = (1ULL << 63) | (1ULL << 61);
            bitboard &rook_bit = get_piece_bitboard(BLACK, ROOK);
            rook_bit ^= mask;
            castling_rights ^= BLACK_KINGSIDE;
            if (castling_rights & BLACK_QUEENSIDE)
            {
                castling_rights ^= BLACK_QUEENSIDE;
            }
            goto play_normal_exit;
        }
    }
    else if (move.has_flag(Move::Flags::QUEEN_CASTLE))
    {
        if (side_to_move == WHITE)
        {
            if (!(castling_rights & WHITE_QUEENSIDE))
            {
                throw std::logic_error("Castling rights missing");
            }
            U64 mask = (1ULL) | (1ULL << 3);
            bitboard &rook_bit = get_piece_bitboard(WHITE, ROOK);
            rook_bit ^= mask;
            if (castling_rights & WHITE_KINGSIDE)
            {
                castling_rights ^= WHITE_KINGSIDE;
            }
            castling_rights ^= WHITE_QUEENSIDE;
            goto play_normal_exit;
        }
        else
        {
            if (!(castling_rights & BLACK_QUEENSIDE))
            {
                throw std::logic_error("Castling rights missing");
            }
            U64 mask = (1ULL << 56) | (1ULL << 59);
            bitboard &rook_bit = get_piece_bitboard(BLACK, ROOK);
            rook_bit ^= mask;
            castling_rights ^= BLACK_QUEENSIDE;
            if (castling_rights & BLACK_KINGSIDE)
            {
                castling_rights ^= BLACK_KINGSIDE;
            }
            goto play_normal_exit;
        }
    }

    // Checking if en passant
    else if (from_piece == PAWN && to_sq == en_passant_sq)
    {
        move.set_flags(Move::EN_PASSANT_CAP);

        // Captures passed pawn
        if (side_to_move == WHITE)
        {
            U64 cap_mask = sq_mask(to_sq - 8);
            get_piece_bitboard(BLACK, PAWN) ^= cap_mask;
        }
        else
        {
            U64 cap_mask = sq_mask(to_sq + 8);
            get_piece_bitboard(WHITE, PAWN) ^= cap_mask;
        }
    }

    // Checking if an opponent square has to be updated
    else if (side_to_move == WHITE && !(occupied_black & to_sq_mask))
    {
        goto play_normal_exit;
    }

    if (side_to_move == BLACK && !(occupied_white & to_sq_mask))
    {
        goto play_normal_exit;
    }

    for (int i{PAWN}; i <= KING; ++i)
    {
        if (get_piece_bitboard(opponent_color, i) & to_sq_mask)
        {
            get_piece_bitboard(opponent_color, i) ^= to_sq_mask;
            move.set_capture(static_cast<Piece>(i));
            goto play_normal_exit;
        }
    }

    throw std::logic_error("Occupancy mask and piece bitboards out of sync");
    return false;
play_normal_exit:
    // Checking if move is pawn double push (if so sets flag)
    if (from_piece == PAWN && abs(from_sq - to_sq) == 16)
    {
        move.set_flags(Move::DOUBLE_PUSH);
        en_passant_sq = (side_to_move == WHITE) ? from_sq + 8 : from_sq - 8;
    }
    else
    {
        en_passant_sq = EN_PASSANT_SQ_NONE;
    }
    update_occupancy();
    switch_trait();
    return true;
}

void Board::unplay(Move move)
{
    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const int from_sq = move.get_from_sq();
    const int to_sq = move.get_to_sq();

    switch_trait();

    const Color color = side_to_move;
    const U64 mask_from_piece = (1ULL << to_sq) | (1ULL << from_sq);
    U64 &from_bitboard = get_piece_bitboard(color, from_piece);
    assert(from_bitboard & (1ULL << to_sq));
    assert(!(from_bitboard & (1ULL << from_sq)));
    from_bitboard ^= mask_from_piece;
    assert(!(from_bitboard & (1ULL << to_sq)));
    assert(from_bitboard & (1ULL << from_sq));
    castling_rights = move.get_prev_castling_rights();
    if (to_piece != NO_PIECE)
    {
        const Color opponent_color = color == WHITE ? BLACK : WHITE;
        get_piece_bitboard(opponent_color, to_piece) |= 1ULL << to_sq;
    }
    switch (move.get_flags())
    {
    case Move::Flags::KING_CASTLE:
        if (color == WHITE)
            get_piece_bitboard(color, ROOK) ^= 0xa0;
        else
            get_piece_bitboard(color, ROOK) ^= 0xa000000000000000;
        break;
    case Move::Flags::QUEEN_CASTLE:
        if (color == WHITE)
            get_piece_bitboard(color, ROOK) ^= 0x9;
        else
            get_piece_bitboard(color, ROOK) ^= 0x900000000000000;
        break;
    case Move::EN_PASSANT_CAP:
        if (color == WHITE)
            get_piece_bitboard(BLACK, PAWN) ^= sq_mask(to_sq - 8);
        else
            get_piece_bitboard(WHITE, PAWN) ^= sq_mask(to_sq + 8);
        en_passant_sq = to_sq;
        break;
    default:
        break;
    }
    update_occupancy();
}