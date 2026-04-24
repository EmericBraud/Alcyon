#ifdef TEXEL_TUNING

#include "engine/eval/pos_eval.hpp"
#include "core/move/generator/move_generator.hpp"

// Transforme un bitboard de pions en un masque où chaque bit
// représente une colonne (file) occupée (8 bits utilisés).
inline uint8_t get_pawn_files(U64 pawns)
{
    pawns |= (pawns >> 32);
    pawns |= (pawns >> 16);
    pawns |= (pawns >> 8);
    return static_cast<uint8_t>(pawns & 0xFF);
}

namespace Eval
{
    static void accumulate_castling_and_safety_features(Color us, const VBoard &board, EvalFeatures &f, double sign)
    {
        const Color them = (Color)!us;
        const int king_sq = board.king_sq[us];
        const int king_file = king_sq & 7;

        const U64 our_pawns = board.get_piece_bitboard(us, PAWN);
        const U64 enemy_pawns = board.get_piece_bitboard(them, PAWN);
        const U64 enemy_heavies = board.get_piece_bitboard(them, ROOK) | board.get_piece_bitboard(them, QUEEN);

        uint8_t our_files = get_pawn_files(our_pawns);
        uint8_t enemy_files = get_pawn_files(enemy_pawns);
        uint8_t vicinity = core::mask::KingVicinityFile[king_file];

        uint8_t open = vicinity & ~our_files & ~enemy_files;
        uint8_t semi_open = vicinity & ~our_files & enemy_files;

        const int open_count = std::popcount(static_cast<uint8_t>(open));
        const int semi_open_count = std::popcount(static_cast<uint8_t>(semi_open));

        U64 open_files_bb = 0;
        U64 semi_files_bb = 0;

        for (int file = king_file - 1; file <= king_file + 1; ++file)
        {
            int64_t valid_file_mask = -static_cast<int64_t>(file >= 0 && file <= 7);
            int clamped_f = file & 7;

            open_files_bb |= (core::mask::File[clamped_f] & -static_cast<int64_t>((open >> clamped_f) & 1)) & valid_file_mask;
            semi_files_bb |= (core::mask::File[clamped_f] & -static_cast<int64_t>((semi_open >> clamped_f) & 1)) & valid_file_mask;
        }

        const int heavy_open_count = std::popcount(enemy_heavies & open_files_bb);
        const int heavy_semi_open_count = std::popcount(enemy_heavies & semi_files_bb);

        f.open_files_near_king += sign * open_count;
        f.semi_open_files_near_king += sign * semi_open_count;
        f.heavy_on_open += sign * heavy_open_count;
        f.heavy_on_semi_open += sign * heavy_semi_open_count;
    }

    static void accumulate_pawn_features(Color color, const VBoard &board, EvalFeatures &f, double sign)
    {
        const U64 our_pawns = board.get_piece_bitboard(color, PAWN);
        const U64 enemy_pawns = board.get_piece_bitboard(!color, PAWN);

        U64 doubled_mask = our_pawns & ((our_pawns >> 8) | (our_pawns >> 16) | (our_pawns >> 24) |
                                        (our_pawns >> 32) | (our_pawns >> 40) | (our_pawns >> 48) | (our_pawns >> 56));
        int num_doubled_files = std::popcount(get_pawn_files(doubled_mask));
        f.doubled_files += sign * num_doubled_files;

        uint8_t files = get_pawn_files(our_pawns);
        uint8_t iso_files = files & ~((files << 1) | (files >> 1));
        int num_iso = std::popcount(iso_files);
        f.isolated_files += sign * num_iso;

        U64 temp_pawns = our_pawns;
        while (temp_pawns)
        {
            const int sq = cpu::pop_lsb(temp_pawns);

            if (!(enemy_pawns & masks.passed[color][sq]))
            {
                const int rank = sq >> 3;
                const int relative_rank = (color == WHITE) ? rank : (7 - rank);

                f.passed_mg[relative_rank] += sign;
                f.passed_eg[relative_rank] += sign;
            }
        }
    }

    static void accumulate_mobility_features(Color us, const VBoard &board, EvalFeatures &f, double sign)
    {
        const U64 occ_all = board.get_occupancy(NO_COLOR);
        const U64 our_occ = board.get_occupancy(us);

        for (int piece = KNIGHT; piece <= QUEEN; ++piece)
        {
            U64 bb = board.get_piece_bitboard(us, piece);

            while (bb)
            {
                const int sq = cpu::pop_lsb(bb);

                U64 moves;
                if (piece == KNIGHT)
                    moves = MoveGen::KnightAttacks[sq];
                else if (piece == BISHOP)
                    moves = MoveGen::generate_bishop_moves(sq, occ_all);
                else if (piece == ROOK)
                    moves = MoveGen::generate_rook_moves(sq, occ_all);
                else
                    moves = MoveGen::generate_bishop_moves(sq, occ_all) | MoveGen::generate_rook_moves(sq, occ_all);

                const int count = std::popcount(moves & ~our_occ);

                if (piece == KNIGHT)
                    f.knight_mob[count] += sign;
                else if (piece == BISHOP)
                    f.bishop_mob[count] += sign;
                else if (piece == ROOK)
                    f.rook_mob[count] += sign;
                else
                    f.queen_mob[std::min(count, 27)] += sign;
            }
        }
    }

    static void accumulate_bishop_pair_features(Color us, const VBoard &board, EvalFeatures &f, double sign)
    {
        if (std::popcount(board.get_piece_bitboard(us, BISHOP)) >= 2)
        {
            f.bishop_pair_mg += sign;
            f.bishop_pair_eg += sign;
        }
    }

    static void accumulate_material_features(Color us, const VBoard &board, EvalFeatures &f, double sign)
    {
        for (int piece = PAWN; piece <= QUEEN; ++piece)
        {
            const int count = std::popcount(board.get_piece_bitboard(us, piece));
            f.material[piece] += sign * count;
        }
    }

    static void accumulate_mopup_features(const VBoard &board, EvalFeatures &f)
    {
        const EvalState &state = board.get_eval_state();

        const double eg_score = (state.eg_pst[WHITE] + state.pieces_val[WHITE]) -
                                (state.eg_pst[BLACK] + state.pieces_val[BLACK]);

        if (std::abs(eg_score) <= 200.0)
            return;

        const Color winner = (eg_score > 0.0) ? WHITE : BLACK;
        const Color loser = (Color)!winner;

        const int wk = board.king_sq[WHITE];
        const int bk = board.king_sq[BLACK];

        const int loser_king = board.king_sq[loser];
        const int k_file = loser_king & 7;
        const int k_rank = loser_king >> 3;

        const int dist_from_center = std::max(std::abs(k_file - 3), std::abs(k_rank - 3));
        const int dist_between_kings = king_distance(wk, bk);

        const double sign = (winner == WHITE) ? 1.0 : -1.0;
        f.king_dist_center += sign * dist_from_center;
        f.king_closeness += sign * (engine_constants::eval::maxDistBetweenKings - dist_between_kings);
    }

    EvalFeatures extract_eval_features(const VBoard &board)
    {
        EvalFeatures f{};

        accumulate_pawn_features(WHITE, board, f, +1.0);
        accumulate_pawn_features(BLACK, board, f, -1.0);

        accumulate_castling_and_safety_features(WHITE, board, f, +1.0);
        accumulate_castling_and_safety_features(BLACK, board, f, -1.0);

        accumulate_bishop_pair_features(WHITE, board, f, +1.0);
        accumulate_bishop_pair_features(BLACK, board, f, -1.0);

        accumulate_mobility_features(WHITE, board, f, +1.0);
        accumulate_mobility_features(BLACK, board, f, -1.0);

        accumulate_material_features(WHITE, board, f, +1.0);
        accumulate_material_features(BLACK, board, f, -1.0);

        accumulate_mopup_features(board, f);

        return f;
    }

    double score_eval_features(const EvalFeatures &f, const VBoard &board)
    {
        const EvalState &state = board.get_eval_state();

        // Base fixe non tunée : material + PST
        double mg = (state.mg_pst[WHITE]) -
                    (state.mg_pst[BLACK]);

        double eg = (state.eg_pst[WHITE]) -
                    (state.eg_pst[BLACK]);

        for (int i = PAWN; i <= QUEEN; ++i)
        {
            mg += f.material[i] * engine_constants::eval::pieces_score[i];
            eg += f.material[i] * engine_constants::eval::pieces_score[i];
        }

        mg += f.doubled_files * engine_constants::eval::doubledFilesMgMalus;
        eg += f.doubled_files * engine_constants::eval::doubledFilesEgMalus;

        mg += f.isolated_files * engine_constants::eval::isolatedFilesMgMalus;
        eg += f.isolated_files * engine_constants::eval::isolatedFilesEgMalus;

        mg += f.open_files_near_king * engine_constants::eval::openFileMalus;
        mg += f.semi_open_files_near_king * engine_constants::eval::semiOpenFileMalus;
        mg += f.heavy_on_open * engine_constants::eval::heavyEnemiesOpenFileMalus;
        mg += f.heavy_on_semi_open * engine_constants::eval::heavyEnemiesSemiOpenFileMalus;

        mg += f.bishop_pair_mg * engine_constants::eval::bishopPairMgBonus;
        eg += f.bishop_pair_eg * engine_constants::eval::bishopPairEgBonus;

        // Mop-up : endgame only
        eg += f.king_dist_center * engine_constants::eval::kingDistFromCenterBonus;
        eg += f.king_closeness * engine_constants::eval::closeKingBonus;

        for (int i = 0; i < 8; ++i)
        {
            mg += f.passed_mg[i] * engine_constants::eval::passed_bonus_mg[i];
            eg += f.passed_eg[i] * engine_constants::eval::passed_bonus_eg[i];
        }

        for (int i = 0; i < 9; ++i)
        {
            mg += f.knight_mob[i] * engine_constants::eval::knight_mob[i];
            eg += f.knight_mob[i] * engine_constants::eval::knight_mob[i];
        }

        for (int i = 0; i < 14; ++i)
        {
            mg += f.bishop_mob[i] * engine_constants::eval::bishop_mob[i];
            eg += f.bishop_mob[i] * engine_constants::eval::bishop_mob[i];
        }

        for (int i = 0; i < 15; ++i)
        {
            mg += f.rook_mob[i] * engine_constants::eval::rook_mob[i];
            eg += f.rook_mob[i] * engine_constants::eval::rook_mob[i];
        }

        for (int i = 0; i < 28; ++i)
        {
            mg += f.queen_mob[i] * engine_constants::eval::queen_mob[i];
            eg += f.queen_mob[i] * engine_constants::eval::queen_mob[i];
        }

        return (mg * state.phase + eg * (engine_constants::eval::totalPhase - state.phase)) /
               engine_constants::eval::totalPhase;
    }
}

#endif