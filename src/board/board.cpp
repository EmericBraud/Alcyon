#include "board.hpp"

#include <iostream>
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
    history.clear();
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

bool Board::play(const Move &move)
{
    if (move.get_from_piece() == PAWN || move.get_to_piece() != NO_PIECE)
    {
        last_irreversible_move_index = history.size(); // On marque ce point
        halfmove_clock = 0;                            // On reset les 50 coups
    }
    else
    {
        halfmove_clock++;
    }
    const UndoInfo info{
        zobrist_key,
        halfmove_clock,
        last_irreversible_move_index,
        move};
    history.push_back(info);
    const int from_sq{move.get_from_sq()}, to_sq{move.get_to_sq()};
    Piece from_piece{move.get_from_piece()};
    const Piece to_piece{move.get_to_piece()};
    if (from_piece == NO_PIECE)
    {
        std::cerr << "No piece found on square" << from_sq << std::endl;
        return false;
    }
    const Color opponent_color = (side_to_move == WHITE) ? BLACK : WHITE;
    assert(is_occupied(from_sq, from_piece, side_to_move));
    assert(!is_occupied(to_sq, NO_PIECE, side_to_move));

    const U64 to_sq_mask{sq_mask(to_sq)};

    zobrist_key ^= zobrist_castling[castling_rights];
    if (en_passant_sq != EN_PASSANT_SQ_NONE)
        zobrist_key ^= zobrist_en_passant[en_passant_sq % 8];
    else
        zobrist_key ^= zobrist_en_passant[8];

    en_passant_sq = EN_PASSANT_SQ_NONE;
    get_piece_bitboard(side_to_move, from_piece) ^= sq_mask(from_sq); // Delete old position
    HASH_PIECE(side_to_move, from_piece, from_sq);

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

    uint8_t required_rights, optional_rights;
    int rook_from_sq, rook_to_sq;
    // Checks if it is a promotion
    switch (move.get_flags())
    {
    case Move::Flags::PROMOTION_MASK:
        from_piece = QUEEN;
        break;
    case Move::Flags::KING_CASTLE:
        required_rights = side_to_move == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        optional_rights = side_to_move == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        rook_from_sq = side_to_move == WHITE ? Square::h1 : Square::h8;
        rook_to_sq = side_to_move == WHITE ? Square::f1 : Square::f8;
        if (!(castling_rights & required_rights))
        {
            throw std::logic_error("Castling rights missing");
        }

        get_piece_bitboard(side_to_move, ROOK) ^= (1ULL << rook_from_sq) | (1ULL << rook_to_sq);
        HASH_PIECE(side_to_move, ROOK, rook_from_sq);
        HASH_PIECE(side_to_move, ROOK, rook_to_sq);
        castling_rights ^= required_rights;
        if (optional_rights & castling_rights)
        {
            castling_rights ^= optional_rights;
        }
        break;
    case Move::Flags::QUEEN_CASTLE:
        required_rights = side_to_move == WHITE ? WHITE_QUEENSIDE : BLACK_QUEENSIDE;
        optional_rights = side_to_move == WHITE ? WHITE_KINGSIDE : BLACK_KINGSIDE;
        rook_from_sq = side_to_move == WHITE ? Square::a1 : Square::a8;
        rook_to_sq = side_to_move == WHITE ? Square::d1 : Square::d8;
        if (!(castling_rights & required_rights))
        {
            throw std::logic_error("Castling rights missing");
        }

        get_piece_bitboard(side_to_move, ROOK) ^= (1ULL << rook_from_sq) | (1ULL << rook_to_sq);
        HASH_PIECE(side_to_move, ROOK, rook_from_sq);
        HASH_PIECE(side_to_move, ROOK, rook_to_sq);
        castling_rights ^= required_rights;
        if (optional_rights & castling_rights)
        {
            castling_rights ^= optional_rights;
        }
        break;
    case Move::Flags::EN_PASSANT_CAP:
        if (side_to_move == WHITE)
        {
            get_piece_bitboard(BLACK, PAWN) ^= sq_mask(to_sq - 8);
            HASH_PIECE(BLACK, PAWN, to_sq - 8);
        }
        else
        {
            get_piece_bitboard(WHITE, PAWN) ^= sq_mask(to_sq + 8);
            HASH_PIECE(WHITE, PAWN, to_sq + 8);
        }
        break;
    case Move::Flags::DOUBLE_PUSH:
        en_passant_sq = (to_sq + from_sq) >> 1; // Avg of from_sq and to_sq gives the en_passant sq
    }

    // If king moves, delete the castling rights
    if (from_piece == KING) // get rid of castling rights if king moved
    {
        if (side_to_move == WHITE)
        {
            castling_rights &= (~WHITE_KINGSIDE);
            castling_rights &= (~WHITE_QUEENSIDE);
        }
        else
        {
            castling_rights &= (~BLACK_KINGSIDE);
            castling_rights &= (~BLACK_QUEENSIDE);
        }
    }

    get_piece_bitboard(side_to_move, from_piece) |= to_sq_mask; // Fill new position
    HASH_PIECE(side_to_move, from_piece, to_sq);

    // Checking if an opponent square has to be updated
    if (to_piece != NO_PIECE)
    {
        get_piece_bitboard(opponent_color, to_piece) ^= to_sq_mask;
        HASH_PIECE(opponent_color, to_piece, to_sq);
    }

    zobrist_key ^= zobrist_castling[castling_rights];
    if (en_passant_sq != EN_PASSANT_SQ_NONE)
        zobrist_key ^= zobrist_en_passant[en_passant_sq % 8];
    else
        zobrist_key ^= zobrist_en_passant[8];

    update_occupancy();
    switch_trait();
    return true;
}

void Board::unplay(const Move move)
{
    const Piece from_piece = (move.get_flags() == Move::PROMOTION_MASK) ? QUEEN : move.get_from_piece();
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
    en_passant_sq = move.get_prev_en_passant(color);

    if (to_piece != NO_PIECE)
    {
        const Color opponent_color = color == WHITE ? BLACK : WHITE;
        get_piece_bitboard(opponent_color, to_piece) |= 1ULL << to_sq;
    }
    switch (move.get_flags())
    {
    case Move::Flags::KING_CASTLE:
        if (color == WHITE)
        {
            get_piece_bitboard(color, ROOK) ^= 0xa0;
        }
        else
        {
            get_piece_bitboard(color, ROOK) ^= 0xa000000000000000;
        }
        break;
    case Move::Flags::QUEEN_CASTLE:
        if (color == WHITE)
        {
            get_piece_bitboard(color, ROOK) ^= 0x9;
        }
        else
        {
            get_piece_bitboard(color, ROOK) ^= 0x900000000000000;
        }
        break;
    case Move::EN_PASSANT_CAP:
        if (color == WHITE)
        {
            get_piece_bitboard(BLACK, PAWN) ^= sq_mask(to_sq - 8);
        }
        else
        {
            get_piece_bitboard(WHITE, PAWN) ^= sq_mask(to_sq + 8);
        }
        break;
    case Move::PROMOTION_MASK:
        get_piece_bitboard(color, PAWN) ^= sq_mask(from_sq);
        get_piece_bitboard(color, QUEEN) ^= sq_mask(from_sq);
        break;
    default:
        break;
    }
    update_occupancy();
    const UndoInfo info = history.back();
    zobrist_key = info.zobrist_key;
    halfmove_clock = info.halfmove_clock;
    last_irreversible_move_index = info.last_irreversible_move_index;
    history.pop_back();
}

bool Board::is_repetition() const
{

    int n = history.size();
    if (n < 4)
        return false;

    int stop = std::max(0, last_irreversible_move_index);

    for (int i = n - 2; i >= stop; i -= 2)
    {
        if (history[i].zobrist_key == zobrist_key)
        {
            return true;
        }
    }
    return false;
}