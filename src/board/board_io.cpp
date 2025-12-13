#include "board.hpp"

#include <iostream>
#include <sstream>

bool Board::load_fen(const std::string_view fen_string)
{
    clear();

    std::string fen(fen_string);
    std::istringstream ss(fen);

    int square{56}; // Starts at 56 (A8)
    std::string token;

    // --- 1. PIECES ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing piece placement." << std::endl;
        return false;
    }

    for (const char c : token)
    {
        if (c == ' ')
        {
            break;
        }

        switch (c)
        {
        // ---- BLACK
        case 'p':
            update_square_bitboard(BLACK, PAWN, square++, true);
            break;
        case 'n':
            update_square_bitboard(BLACK, KNIGHT, square++, true);
            break;
        case 'b':
            update_square_bitboard(BLACK, BISHOP, square++, true);
            break;
        case 'r':
            update_square_bitboard(BLACK, ROOK, square++, true);
            break;
        case 'q':
            update_square_bitboard(BLACK, QUEEN, square++, true);
            break;
        case 'k':
            update_square_bitboard(BLACK, KING, square++, true);
            break;

        // ---- WHITE
        case 'P':
            update_square_bitboard(WHITE, PAWN, square++, true);
            break;
        case 'N':
            update_square_bitboard(WHITE, KNIGHT, square++, true);
            break;
        case 'B':
            update_square_bitboard(WHITE, BISHOP, square++, true);
            break;
        case 'R':
            update_square_bitboard(WHITE, ROOK, square++, true);
            break;
        case 'Q':
            update_square_bitboard(WHITE, QUEEN, square++, true);
            break;
        case 'K':
            update_square_bitboard(WHITE, KING, square++, true);
            break;

        // Other
        case '/':
            square -= 16;
            break;

        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        {
            int jump{c - '0'};
            square += jump;
        }
        break;

        default:
            std::cerr << "WARNING: Invalid character in piece placement: " << c << std::endl;
            return false;
        }
    }

    // --- 2. SIDE TO MOVE ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing side to move." << std::endl;
        return false;
    }
    if (token.length() != 1)
    {
        std::cerr << "WARNING: Invalid side to move field." << std::endl;
        return false;
    }

    char side_char = token[0];
    if (side_char == 'w')
    {
        side_to_move = WHITE;
    }
    else if (side_char == 'b')
    {
        side_to_move = BLACK;
    }
    else
    {
        std::cerr << "WARNING: Invalid side to move: " << side_char << std::endl;
        return false;
    }

    // --- 3. CASTLING RIGHTS ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing castling rights." << std::endl;
        return false;
    }
    // Assumons que castling_rights a été initialisé à 0 par clear()
    for (const char c : token)
    {
        switch (c)
        {
        case 'K':
            castling_rights |= WHITE_KINGSIDE;
            break;
        case 'Q':
            castling_rights |= WHITE_QUEENSIDE;
            break;
        case 'k':
            castling_rights |= BLACK_KINGSIDE;
            break;
        case 'q':
            castling_rights |= BLACK_QUEENSIDE;
            break;
        case '-': // No castling rights
            if (token.length() > 1)
            {
                std::cerr << "WARNING: Invalid castling field: -" << std::endl;
                return false;
            }
            break;
        default:
            std::cerr << "WARNING: Invalid character in castling rights: " << c << std::endl;
            return false;
        }
    }

    // --- 4. EN PASSANT TARGET SQUARE ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing en passant target." << std::endl;
        return false;
    }
    if (token == "-")
    {
        en_passant_sq = EN_PASSANT_SQ_NONE;
    }
    else if (token.length() == 2)
    {
        // Convert format a3 -> index (0-63)
        // Simple example: 'a' -> 0, '1' -> 0. (column * 8 + row)
        int file = token[0] - 'a';
        int rank = token[1] - '1';
        en_passant_sq = (rank * 8) + file;
    }
    else
    {
        std::cerr << "WARNING: Invalid en passant field: " << token << std::endl;
        return false;
    }

    // --- 5. HALFMOVE CLOCK (50 moves rule) ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing halfmove clock." << std::endl;
        return false;
    }
    try
    {
        halfmove_clock = std::stoi(token);
    }
    catch (const std::exception &e)
    {
        std::cerr << "WARNING: Invalid halfmove clock value: " << token << std::endl;
        return false;
    }

    // --- 6. FULLMOVE NUMBER ---
    if (!std::getline(ss, token, ' '))
    {
        std::cerr << "WARNING: FEN incomplete - missing fullmove number." << std::endl;
        return false;
    }
    try
    {
        fullmove_number = std::stoi(token);
    }
    catch (const std::exception &e)
    {
        std::cerr << "WARNING: Invalid fullmove number value: " << token << std::endl;
        return false;
    }

    // Update occupancy masks
    update_occupancy();
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
    std::cout << "Side to Move: " << (side_to_move == WHITE ? "White (w)" : "Black (b)") << std::endl;

    // Display castling rights using binary mask check
    std::cout << "Castling Rights: " << (castling_rights & WHITE_KINGSIDE ? 'K' : '-')
              << (castling_rights & WHITE_QUEENSIDE ? 'Q' : '-')
              << (castling_rights & BLACK_KINGSIDE ? 'k' : '-')
              << (castling_rights & BLACK_QUEENSIDE ? 'q' : '-') << std::endl;

    // Display en passant square
    std::cout << "En Passant Square: " << (en_passant_sq == 0 ? "-" : std::to_string(en_passant_sq)) << std::endl;
}