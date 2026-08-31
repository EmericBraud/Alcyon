#include "gpu_position.hpp"

#include <bit>

namespace gpu_eval {

namespace {
char piece_char(Piece piece, Color color) {
    static constexpr char kWhiteChars[6] = {'P', 'N', 'B', 'R', 'Q', 'K'};
    static constexpr char kBlackChars[6] = {'p', 'n', 'b', 'r', 'q', 'k'};
    return color == WHITE ? kWhiteChars[piece] : kBlackChars[piece];
}
} // namespace

std::string GpuPosition::to_fen() const {
    std::string fen;
    fen.reserve(80);

    for (int rank = 7; rank >= 0; --rank) {
        int empty_run = 0;
        for (int file = 0; file < 8; ++file) {
            const int sq = rank * 8 + file;
            Piece found_piece = NO_PIECE;
            Color found_color = NO_COLOR;
            for (int p = 0; p < constants::PieceTypeCount && found_piece == NO_PIECE; ++p) {
                if (core::mask::is_set(pieces_occ[get_piece_index(static_cast<Piece>(p), WHITE)], sq)) {
                    found_piece = static_cast<Piece>(p);
                    found_color = WHITE;
                } else if (core::mask::is_set(pieces_occ[get_piece_index(static_cast<Piece>(p), BLACK)], sq)) {
                    found_piece = static_cast<Piece>(p);
                    found_color = BLACK;
                }
            }
            if (found_piece == NO_PIECE) {
                ++empty_run;
                continue;
            }
            if (empty_run > 0) {
                fen += std::to_string(empty_run);
                empty_run = 0;
            }
            fen += piece_char(found_piece, found_color);
        }
        if (empty_run > 0) {
            fen += std::to_string(empty_run);
        }
        if (rank > 0) {
            fen += '/';
        }
    }

    fen += ' ';
    fen += (side_to_move == WHITE ? 'w' : 'b');

    fen += ' ';
    std::string castling;
    if (castling_rights & WHITE_KINGSIDE) castling += 'K';
    if (castling_rights & WHITE_QUEENSIDE) castling += 'Q';
    if (castling_rights & BLACK_KINGSIDE) castling += 'k';
    if (castling_rights & BLACK_QUEENSIDE) castling += 'q';
    fen += castling.empty() ? "-" : castling;

    fen += ' ';
    if (en_passant_sq == constants::EnPassantSqNone) {
        fen += '-';
    } else {
        const int file = en_passant_sq & 7;
        const int rank = en_passant_sq >> 3;
        fen += static_cast<char>('a' + file);
        fen += static_cast<char>('1' + rank);
    }

    fen += ' ';
    fen += std::to_string(halfmove_clock);
    fen += " 1"; // fullmove number -- unused by the encoder/eval, placeholder.

    return fen;
}

} // namespace gpu_eval
