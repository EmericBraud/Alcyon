#pragma once

#include <string_view>
#include <array>
#include <cstdint>
#include <mutex>

#include "common/constants.hpp"
#include "common/logger.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "engine/config/config.hpp"

#ifndef TB_NO_HELPER_API
#define TB_NO_HELPER_API
#endif
extern "C"
{
#include "tbprobe.h"
}

class TableBase
{
    static constexpr const char *TablebaseFolderPath = DATA_PATH "syzygy";
    struct RootRawResult
    {
        bool success;
        unsigned wdl;
        unsigned from;
        unsigned to;
        unsigned promotes;
        unsigned ep;
        unsigned dtz;
    };
    std::mutex tb_mutex;

    RootRawResult probe_root_impl(const Board &board)
    {
        const int ep_sq = board.get_en_passant_sq() == constants::EnPassantSqNone ? 0 : board.get_en_passant_sq();
        unsigned probe_result = tb_probe_root(board.get_occupancy(WHITE),
                                              board.get_occupancy(BLACK),
                                              board.get_piece_bitboard<Color::NO_COLOR, KING>(),
                                              board.get_piece_bitboard<Color::NO_COLOR, QUEEN>(),
                                              board.get_piece_bitboard<Color::NO_COLOR, ROOK>(),
                                              board.get_piece_bitboard<Color::NO_COLOR, BISHOP>(),
                                              board.get_piece_bitboard<Color::NO_COLOR, KNIGHT>(),
                                              board.get_piece_bitboard<Color::NO_COLOR, PAWN>(),
                                              board.get_halfmove_clock(),
                                              board.get_castling_rights(),
                                              ep_sq,
                                              !board.get_side_to_move(),
                                              NULL);
        RootRawResult r;
        if (probe_result == TB_RESULT_FAILED || probe_result == TB_RESULT_STALEMATE || probe_result == TB_RESULT_CHECKMATE)
        {
            RootRawResult r{};
            r.success = false;
            return r;
        }
        r.success = true;
        r.wdl = TB_GET_WDL(probe_result);
        r.from = TB_GET_FROM(probe_result);
        r.to = TB_GET_TO(probe_result);
        r.promotes = TB_GET_PROMOTES(probe_result);
        r.ep = TB_GET_EP(probe_result);
        r.dtz = TB_GET_DTZ(probe_result);
        return r;
    }

public:
    static constexpr int MaxNumTableBasePieces = 5;

    struct RootResult
    {
        Move move;
        int score;
    };
    enum class WDL_Result : unsigned
    {
        LOSS = TB_LOSS,
        BLESSED_LOSS = TB_BLESSED_LOSS,
        DRAW = TB_DRAW,
        CURSED_WIN = TB_CURSED_WIN,
        WIN = TB_WIN,
        FAIL = TB_RESULT_FAILED

    };

    TableBase()
    {
        if (!tb_init(TablebaseFolderPath))
            logs::error << "TB init failed" << std::endl;

        else if (TB_LARGEST == 0)
            logs::error << "No TB found" << std::endl;
        logs::debug << "TB initialized" << std::endl;
    }

    ~TableBase()
    {
        tb_free();
    }

    // This function should be called inside negamax
    WDL_Result probe_wdl(const Board &board)
    {
        const int ep_sq = board.get_en_passant_sq() == constants::EnPassantSqNone ? 0 : board.get_en_passant_sq();
        std::lock_guard<std::mutex> lock(tb_mutex);
        return static_cast<WDL_Result>(tb_probe_wdl(board.get_occupancy(WHITE),
                                                    board.get_occupancy(BLACK),
                                                    board.get_piece_bitboard<Color::NO_COLOR, KING>(),
                                                    board.get_piece_bitboard<Color::NO_COLOR, QUEEN>(),
                                                    board.get_piece_bitboard<Color::NO_COLOR, ROOK>(),
                                                    board.get_piece_bitboard<Color::NO_COLOR, BISHOP>(),
                                                    board.get_piece_bitboard<Color::NO_COLOR, KNIGHT>(),
                                                    board.get_piece_bitboard<Color::NO_COLOR, PAWN>(),
                                                    board.get_halfmove_clock(),
                                                    board.get_castling_rights(),
                                                    ep_sq,
                                                    !board.get_side_to_move()));
    }

    RootResult probe_root(const Board &board)
    {
        RootRawResult r_raw = probe_root_impl(board);
        RootResult r;
        if (!r_raw.success)
        {
            r.move = 0;
            r.score = 0;
            return r;
        }
        Piece promo_piece;
        switch (r_raw.promotes)
        {
        case TB_PROMOTES_QUEEN:
            promo_piece = QUEEN;
            break;
        case TB_PROMOTES_ROOK:
            promo_piece = ROOK;
            break;
        case TB_PROMOTES_BISHOP:
            promo_piece = BISHOP;
            break;
        case TB_PROMOTES_KNIGHT:
            promo_piece = KNIGHT;
            break;
        default:
        case TB_PROMOTES_NONE:
            promo_piece = NO_PIECE;
            break;
        }
        Move m{
            static_cast<int>(r_raw.from),
            static_cast<int>(r_raw.to),
            board.get_p(r_raw.from),
            Move::NONE,
            board.get_p(r_raw.to),
            promo_piece};
        MoveGen::init_move_flags(board, m);
        r.move = m;
        Color us = board.get_side_to_move();
        Color winning_color;
        switch (r_raw.wdl)
        {
        case TB_LOSS:
            winning_color = !us;
            break;
        case TB_BLESSED_LOSS:
        case TB_DRAW:
        case TB_CURSED_WIN:
            winning_color = NO_COLOR;
            break;
        case TB_WIN:
            winning_color = us;
            break;
        case TB_RESULT_FAILED:
        default:
            return {0, 0};
        }

        if (winning_color == NO_COLOR)
        {
            r.score = 0;
            return r;
        }
        r.score = engine_constants::eval::SyzygyScore - r_raw.dtz;
        if (winning_color != us)
            r.score = -r.score;
        return r;
    }
};