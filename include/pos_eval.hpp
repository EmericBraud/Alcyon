#pragma once

#include "move_generator.hpp"

namespace eval
{
    static const int pawnPhase = 0;
    static const int knightPhase = 1;
    static const int bishopPhase = 1;
    static const int rookPhase = 2;
    static const int queenPhase = 4;
    static const int totalPhase = 24;

    static constexpr int mg_pawn_table[64] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0};

    static constexpr int mg_knight_table[64] = {
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50};

    static constexpr int mg_bishop_table[64] = {
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20};

    static constexpr int mg_rook_table[64] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0};

    static constexpr int mg_queen_table[64] = {
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20};

    static constexpr int mg_king_table[64] = {
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30};

    static constexpr int eg_king_table[64] = {
        -50, -40, -30, -20, -20, -30, -40, -50,
        -30, -20, -10, 0, 0, -10, -20, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 30, 40, 40, 30, -10, -30,
        -30, -10, 20, 30, 30, 20, -10, -30,
        -30, -30, 0, 0, 0, 0, -30, -30,
        -50, -30, -30, -30, -30, -30, -30, -50};

    static constexpr std::array<int, N_PIECES_TYPE_HALF> pieces_score = {
        100, 300, 300, 500, 900, 10000};

    // Table de correspondance pour le milieu de jeu
    static const int *mg_tables[] = {
        mg_pawn_table, mg_knight_table, mg_bishop_table, mg_rook_table, mg_queen_table, mg_king_table};

    static constexpr int knight_mob[9] = {
        -20, -10, 0, 5, 10, 15, 20, 25, 30};

    static constexpr int bishop_mob[14] = {
        -20, -10, 0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65};

    static constexpr int rook_mob[15] = {
        -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55};

    static constexpr int queen_mob[28] = {
        -20, -15, -10, -5, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
        22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46};

    struct PawnMasks
    {
        std::array<bitboard, 8> file;
        std::array<bitboard, 8> adjacent;
        std::array<std::array<bitboard, 64>, 2> passed;
    };
    constexpr PawnMasks generate_masks()
    {
        PawnMasks m = {};

        // 1. Colonnes et colonnes adjacentes
        for (int f = 0; f < 8; f++)
        {
            m.file[f] = 0x0101010101010101ULL << f;
        }
        for (int f = 0; f < 8; f++)
        {
            if (f > 0)
                m.adjacent[f] |= m.file[f - 1];
            if (f < 7)
                m.adjacent[f] |= m.file[f + 1];
        }

        // 2. Pions passés
        for (int sq = 0; sq < 64; sq++)
        {
            int f = sq % 8;
            int r = sq / 8;

            for (int r_target = 0; r_target < 8; r_target++)
            {
                for (int f_target = 0; f_target < 8; f_target++)
                {
                    // Si la colonne cible est f-1, f, ou f+1
                    if (f_target >= f - 1 && f_target <= f + 1)
                    {
                        int target_sq = r_target * 8 + f_target;
                        if (r_target > r)
                            m.passed[0][sq] |= (1ULL << target_sq); // WHITE
                        if (r_target < r)
                            m.passed[1][sq] |= (1ULL << target_sq); // BLACK
                    }
                }
            }
        }
        return m;
    }

    static constexpr PawnMasks masks = generate_masks();

    inline int evaluate_castling_and_safety(Color color, const Board &board)
    {
        int score = 0;
        int king_sq = std::countr_zero(board.get_piece_bitboard(color, KING));

        // 1. Bonus pour le roque effectué (Heuristique simple)
        // Si le roi est en g1/c1 (blanc) ou g8/c8 (noir), on assume souvent le roque
        bool has_castled = (color == WHITE) ? (king_sq == Square::g1 || king_sq == Square::c1)
                                            : (king_sq == Square::g8 || king_sq == Square::c8);
        if (has_castled)
            score += 40;

        // 2. Malus si on a perdu les droits au roque sans roquer
        if (!has_castled && !(color == WHITE ? (board.get_castling_rights() & (CastlingRights::WHITE_KINGSIDE | CastlingRights::WHITE_QUEENSIDE)) : (board.get_castling_rights() & (CastlingRights::BLACK_KINGSIDE | CastlingRights::BLACK_QUEENSIDE))))
        {
            score -= 50;
        }

        // 3. Bouclier de pions (Amélioration de ton King Safety)
        // Pénalité progressive au lieu d'un simple -50
        bitboard pawns = board.get_piece_bitboard(color, PAWN);
        bitboard shield_mask = (color == WHITE) ? (0x700ULL << 8) : (0x700ULL << 40); // Simplifié pour g1/g8
        // Il faudrait ajuster le masque selon la colonne du Roi (Aile Roi ou Aile Dame)

        int pawn_count = std::popcount(pawns & shield_mask);
        if (pawn_count == 3)
            score += 20;
        else if (pawn_count == 0)
            score -= 80; // Très dangereux

        return score;
    }

    inline void evaluate_pawns(Color color, const Board &board, int &mg, int &eg)
    {
        bitboard our_pawns = board.get_piece_bitboard(color, PAWN);
        bitboard enemy_pawns = board.get_piece_bitboard(!color, PAWN);
        bitboard temp_pawns = our_pawns;

        while (temp_pawns)
        {
            int sq = std::countr_zero(temp_pawns);
            int file = sq % 8;
            int relative_rank = (color == WHITE) ? (sq / 8) : (7 - (sq / 8));

            // 1. Pions doublés
            if (std::popcount(our_pawns & masks.file[file]) > 1)
            {
                mg -= 15;
                eg -= 20;
            }

            // 2. Pions isolés
            if (!(our_pawns & masks.adjacent[file]))
            {
                mg -= 20;
                eg -= 25;
            }

            // 3. Pions passés
            // On utilise masks.passed[0] pour blanc et [1] pour noir
            if (!(enemy_pawns & masks.passed[color][sq]))
            {
                static constexpr int passed_bonus_mg[] = {0, 5, 10, 20, 40, 70, 120, 0};
                static constexpr int passed_bonus_eg[] = {0, 10, 20, 40, 80, 150, 250, 0};
                mg += passed_bonus_mg[relative_rank];
                eg += passed_bonus_eg[relative_rank];
            }
            temp_pawns &= (temp_pawns - 1);
        }
    }

    inline int eval(const Board &board)
    {
        int mg[2] = {0, 0};
        int eg[2] = {0, 0};
        int gamePhase = 0;

        for (int color = WHITE; color <= BLACK; ++color)
        {
            // Évaluation des Pions (Structure + Passés)
            evaluate_pawns((Color)color, board, mg[color], eg[color]);

            for (int piece = PAWN; piece <= KING; ++piece)
            {
                bitboard bb = board.get_piece_bitboard((Color)color, piece);
                const int *mg_table = mg_tables[piece];

                while (bb)
                {
                    int sq = std::countr_zero(bb);
                    int mirror_sq = (color == WHITE) ? sq : sq ^ 56;

                    // Matériel et PST
                    mg[color] += pieces_score[piece] + mg_table[mirror_sq];
                    eg[color] += pieces_score[piece] + (piece == KING ? eg_king_table[mirror_sq] : mg_table[mirror_sq]);

                    // Calcul de la Phase (Pour l'interpolation)
                    if (piece != PAWN && piece != KING)
                    {
                        if (piece == KNIGHT)
                            gamePhase += knightPhase;
                        else if (piece == BISHOP)
                            gamePhase += bishopPhase;
                        else if (piece == ROOK)
                            gamePhase += rookPhase;
                        else if (piece == QUEEN)
                            gamePhase += queenPhase;
                    }

                    // Mobilité
                    if (piece > PAWN && piece < KING)
                    {
                        U64 mobility_mask;
                        if (piece == KNIGHT)
                            mobility_mask = MoveGen::KnightAttacks[sq];
                        else if (piece == BISHOP)
                            mobility_mask = MoveGen::generate_bishop_moves(sq, board);
                        else if (piece == ROOK)
                            mobility_mask = MoveGen::generate_rook_moves(sq, board);
                        else
                            mobility_mask = MoveGen::generate_bishop_moves(sq, board) | MoveGen::generate_rook_moves(sq, board);

                        int count = std::popcount(mobility_mask & ~board.get_occupancy((Color)color));
                        int bonus = 0;
                        if (piece == KNIGHT)
                            bonus = knight_mob[count];
                        else if (piece == BISHOP)
                            bonus = bishop_mob[count];
                        else if (piece == ROOK)
                            bonus = rook_mob[count];
                        else
                            bonus = queen_mob[count];

                        mg[color] += bonus;
                        eg[color] += bonus;
                    }
                    bb &= (bb - 1);
                }
            }
            // Sécurité et Roques
            mg[color] += evaluate_castling_and_safety((Color)color, board);
        }

        int mgScore = mg[WHITE] - mg[BLACK];
        int egScore = eg[WHITE] - eg[BLACK];

        // Interpolation
        int phase = std::min(gamePhase, totalPhase);
        return (mgScore * phase + egScore * (totalPhase - phase)) / totalPhase;
    }

    inline int eval_relative(Color side_to_move, const Board &board)
    {
        int score = eval(board);
        return (side_to_move == WHITE) ? score : -score;
    }

    inline int get_piece_score(int piece)
    {
        return pieces_score[piece];
    }
}