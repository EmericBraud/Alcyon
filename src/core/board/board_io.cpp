#include "core/board.hpp"

#include <iostream>
#include <sstream>
bool Board::load_fen(const std::string_view fen_string)
{
    clear();
    init_zobrist();

    std::string fen(fen_string);
    std::istringstream ss(fen);
    int square{56};
    std::string token;

    if (!std::getline(ss, token, ' '))
        return false;

    for (const char c : token)
    {
        if (c == ' ')
            break;

        // Gestion des pièces
        if (std::isalpha(c))
        {
            Color color = std::isupper(c) ? WHITE : BLACK;
            Piece type;
            char upper_c = std::toupper(c);

            switch (upper_c)
            {
            case 'P':
                type = PAWN;
                break;
            case 'N':
                type = KNIGHT;
                break;
            case 'B':
                type = BISHOP;
                break;
            case 'R':
                type = ROOK;
                break;
            case 'Q':
                type = QUEEN;
                break;
            case 'K':
                type = KING;
                break;
            default:
                return false;
            }

            // Mise à jour Mailbox unifiée
            mailbox[square] = (color << COLOR_SHIFT) | type;
            update_square_bitboard(color, type, square++, true);
        }
        else if (c == '/')
        {
            square -= 16;
        }
        else if (std::isdigit(c))
        {
            square += (c - '0');
        }
    }

    // --- SIDE TO MOVE ---
    if (!std::getline(ss, token, ' '))
        return false;
    state.side_to_move = (token[0] == 'w') ? WHITE : BLACK;

    // --- CASTLING RIGHTS ---
    if (!std::getline(ss, token, ' '))
        return false;
    for (const char c : token)
    {
        if (c == 'K')
            state.castling_rights |= WHITE_KINGSIDE;
        else if (c == 'Q')
            state.castling_rights |= WHITE_QUEENSIDE;
        else if (c == 'k')
            state.castling_rights |= BLACK_KINGSIDE;
        else if (c == 'q')
            state.castling_rights |= BLACK_QUEENSIDE;
    }

    // --- EN PASSANT ---
    if (!std::getline(ss, token, ' '))
        return false;
    if (token == "-")
        state.en_passant_sq = EN_PASSANT_SQ_NONE;
    else
    {
        int file = token[0] - 'a';
        int rank = token[1] - '1';
        state.en_passant_sq = (rank * 8) + file;
    }

    // --- CLOCKS ---
    if (std::getline(ss, token, ' '))
        state.halfmove_clock = std::stoi(token);

    update_occupancy();
    compute_full_hash();
    eval_state = EvalState(pieces_occ);
    return true;
}

char Board::piece_to_char(Color color, Piece type) const
{
    if (type == NO_PIECE)
    {
        return ' '; // Empty square
    }

    // Array holding piece characters (Index 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K)
    // We use a simple array as the type enum matches the index
    const char piece_chars[] = {'P', 'N', 'B', 'R', 'Q', 'K', ' '};

    char result = piece_chars[type];

    if (color == BLACK)
    {
        // Convert to lowercase for black pieces
        return std::tolower(static_cast<unsigned char>(result));
    }
    else
    {
        // White pieces are already uppercase
        return result;
    }
}

void Board::show() const
{
    std::cout << "\n   +---------------------------------------+" << std::endl;

    // Iterates from rank 8 (index 7) down to rank 1 (index 0)
    for (int rank = 7; rank >= 0; --rank)
    {
        std::cout << " " << (rank + 1) << " |"; // Display rank number

        for (int file = 0; file < 8; ++file)
        {
            int square_index = rank * 8 + file;

            // Get piece at square
            auto [color, piece_type] = get_piece_on_square(square_index);

            // Convert to character (e.g., 'p', 'K', ' ')
            char piece_char = piece_to_char(color, piece_type);

            std::cout << " " << piece_char << " ";

            // Optional separator bar
            if (file < 7)
            {
                std::cout << " |";
            }
        }

        std::cout << " |" << std::endl;

        if (rank > 0)
        {
            std::cout << "   +---------------------------------------+" << std::endl;
        }
    }

    std::cout << "   +---------------------------------------+" << std::endl;
    // Display file letters (A-H)
    std::cout << "      A    B    C    D    E    F    G    H\n"
              << std::endl;

    // Display basic state info
    std::cout << "Side to Move: " << (state.side_to_move == WHITE ? "White (w)" : "Black (b)") << std::endl;

    // Display castling rights using binary mask check
    std::cout << "Castling Rights: " << (state.castling_rights & WHITE_KINGSIDE ? 'K' : '-')
              << (state.castling_rights & WHITE_QUEENSIDE ? 'Q' : '-')
              << (state.castling_rights & BLACK_KINGSIDE ? 'k' : '-')
              << (state.castling_rights & BLACK_QUEENSIDE ? 'q' : '-') << std::endl;

    // Display en passant square
    std::cout << "En Passant Square: " << (state.en_passant_sq == 0 ? "-" : std::to_string(state.en_passant_sq)) << std::endl;
}

void Board::compute_full_hash()
{
    zobrist_key = 0;

    for (int sq = 0; sq < 64; ++sq)
    {
        PieceInfo p = get_piece_on_square(sq);
        if (p.second != NO_PIECE)
        {
            int piece_index = p.second + N_PIECES_TYPE_HALF * p.first;
            zobrist_key ^= zobrist_table[piece_index][sq];
        }
    }

    if (get_side_to_move() == BLACK)
    {
        zobrist_key ^= zobrist_black_to_move;
    }

    zobrist_key ^= zobrist_castling[get_castling_rights()];

    if (get_en_passant_sq() != EN_PASSANT_SQ_NONE)
    {
        zobrist_key ^= zobrist_en_passant[get_en_passant_sq() % 8];
    }
    else
    {
        zobrist_key ^= zobrist_en_passant[8];
    }
}

void Board::undo_last_move()
{
    if (get_history()->empty())
        return;
    unplay(get_history()->back().move);
}