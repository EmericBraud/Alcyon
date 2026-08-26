#include "plane_batch.h"

#include "chess.h"

namespace chess26::cnn {

using chess::CastlingRights;
using chess::Color;
using chess::Piece;
using chess::PieceType;
using chess::Position;
using chess::Square;

namespace {

// Flips the rank while keeping the file — standard trick since a
// Square's id packs as (rank << 3) | file, see chess.h. Used to
// re-orient the board so the side to move is always "at the bottom",
// the same convention AlphaZero-style nets use.
constexpr int flip_rank(int square_index) { return square_index ^ 56; }

int plane_for_piece(PieceType type, bool is_us) {
    // PLANE_US_* / PLANE_THEM_* are laid out in PieceType ordinal
    // order (Pawn..King), offset by 6 for "them" — see plane_batch.h.
    const int ordinal = static_cast<int>(type);
    return is_us ? PLANE_US_PAWN + ordinal : PLANE_THEM_PAWN + ordinal;
}

}  // namespace

PlaneBatch::PlaneBatch(const std::vector<binpack::TrainingDataEntry>& entries)
    : size(static_cast<int>(entries.size())) {
    planes = new float[static_cast<std::size_t>(size) * NUM_PLANES * PLANE_SIZE]();
    score = new float[size];
    result = new float[size];
    piece_count = new int[size];

    for (int i = 0; i < size; ++i) {
        fill_entry(i, entries[i]);
    }
}

PlaneBatch::~PlaneBatch() {
    delete[] planes;
    delete[] score;
    delete[] result;
    delete[] piece_count;
}

void PlaneBatch::fill_entry(int i, const binpack::TrainingDataEntry& e) {
    const Position& pos = e.pos;
    const Color us = pos.sideToMove();
    const Color them = (us == Color::White) ? Color::Black : Color::White;
    const bool orient_flip = (us == Color::Black);

    float* base = planes + static_cast<std::size_t>(i) * NUM_PLANES * PLANE_SIZE;
    const auto set = [&](int plane, int square_index) {
        base[static_cast<std::size_t>(plane) * PLANE_SIZE + square_index] = 1.0f;
    };

    int non_king_pieces = 0;
    for (int sq = 0; sq < 64; ++sq) {
        const Piece piece = pos.pieceAt(Square(sq));
        if (piece == Piece::none()) continue;

        const bool is_us = piece.color() == us;
        const int plane = plane_for_piece(piece.type(), is_us);
        const int oriented_sq = orient_flip ? flip_rank(sq) : sq;
        set(plane, oriented_sq);

        if (piece.type() != PieceType::King) ++non_king_pieces;
    }
    piece_count[i] = non_king_pieces;

    // Side to move (constant plane, matches NNUE-style "us is always
    // white" reorientation semantics: after flipping, this records
    // the *actual* color to move for any asymmetric rule the net
    // might need, e.g. none currently, kept for future-proofing).
    if (us == Color::White) {
        for (int sq = 0; sq < 64; ++sq) set(PLANE_SIDE_TO_MOVE_IS_WHITE, sq);
    }

    const CastlingRights rights = pos.castlingRights();
    const CastlingRights us_king = (us == Color::White) ? CastlingRights::WhiteKingSide
                                                          : CastlingRights::BlackKingSide;
    const CastlingRights us_queen = (us == Color::White) ? CastlingRights::WhiteQueenSide
                                                           : CastlingRights::BlackQueenSide;
    const CastlingRights them_king = (them == Color::White) ? CastlingRights::WhiteKingSide
                                                              : CastlingRights::BlackKingSide;
    const CastlingRights them_queen = (them == Color::White) ? CastlingRights::WhiteQueenSide
                                                               : CastlingRights::BlackQueenSide;

    if (chess::contains(rights, us_king)) {
        for (int sq = 0; sq < 64; ++sq) set(PLANE_US_CASTLE_KINGSIDE, sq);
    }
    if (chess::contains(rights, us_queen)) {
        for (int sq = 0; sq < 64; ++sq) set(PLANE_US_CASTLE_QUEENSIDE, sq);
    }
    if (chess::contains(rights, them_king)) {
        for (int sq = 0; sq < 64; ++sq) set(PLANE_THEM_CASTLE_KINGSIDE, sq);
    }
    if (chess::contains(rights, them_queen)) {
        for (int sq = 0; sq < 64; ++sq) set(PLANE_THEM_CASTLE_QUEENSIDE, sq);
    }

    const Square ep = pos.epSquare();
    if (ep != Square::none()) {
        const int oriented_ep = orient_flip ? flip_rank(static_cast<int>(ep)) : static_cast<int>(ep);
        set(PLANE_EN_PASSANT, oriented_ep);
    }

    // Normalized rule50 counter, broadcast over the whole plane —
    // same convention as the castling/side-to-move constant planes.
    const float rule50_normalized =
        static_cast<float>(pos.rule50Counter()) / 100.0f;
    for (int sq = 0; sq < 64; ++sq) {
        base[static_cast<std::size_t>(PLANE_RULE50) * PLANE_SIZE + sq] = rule50_normalized;
    }

    for (int sq = 0; sq < 64; ++sq) {
        const int oriented_sq = orient_flip ? flip_rank(sq) : sq;
        if (pos.isSquareAttacked(Square(sq), us)) set(PLANE_US_ATTACKS, oriented_sq);
        if (pos.isSquareAttacked(Square(sq), them)) set(PLANE_THEM_ATTACKS, oriented_sq);
    }

    score[i] = static_cast<float>(e.score);
    result[i] = static_cast<float>(e.result);
}

std::uint64_t hash_position(const Position& pos) {
    // 24-byte stable serialization (8-byte occupied bitboard + 16-byte
    // packed piece state) — see CompressedPosition::writeToBigEndian in
    // chess.h. Hashed with FNV-1a, a plain deterministic hash with no
    // dependency on std::hash's per-process seeding.
    unsigned char buf[24];
    pos.compress().writeToBigEndian(buf);

    std::uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
    for (unsigned char byte : buf) {
        h ^= byte;
        h *= 0x100000001b3ULL;  // FNV-1a 64-bit prime
    }
    return h;
}

}  // namespace chess26::cnn
