#pragma once

#include "common/constants.hpp"
#include "common/cpu.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "core/move/move.hpp"
#include "engine/config/eval.hpp"
#include "core/board/zobrist.hpp"

struct EvalState
{
    int16_t mg_pst[2];  // Score Middle-Game PST
    int16_t eg_pst[2];  // Score End-Game PST
    int8_t phase;       // Phase actuelle (0 à 24)
    U64 pawn_key;       // Clé Zobrist spécifique aux pions (pour la Pawn Table)
    uint8_t king_sq[2]; // Position des rois
    int16_t pieces_val[2];

    EvalState() = default;
    EvalState(const std::array<U64, constants::NumPieceVariants> occupancy)
    {
        mg_pst[0] = mg_pst[1] = eg_pst[0] = eg_pst[1] = 0;
        pieces_val[0] = pieces_val[1] = 0;
        phase = 0;
        pawn_key = 0;

        for (int i = WHITE; i <= BLACK; ++i)
        {
            for (int j = PAWN; j <= KING; ++j)
            {
                U64 mask = occupancy[j + i * constants::PieceTypeCount];
                while (mask)
                {
                    int sq = cpu::pop_lsb(mask);
                    add_piece(static_cast<Piece>(j), sq, static_cast<Color>(i));

                    if (j != PAWN && j != KING)
                    {
                        if (j == KNIGHT)
                            phase += engine::config::eval::knightPhase;
                        else if (j == BISHOP)
                            phase += engine::config::eval::bishopPhase;
                        else if (j == ROOK)
                            phase += engine::config::eval::rookPhase;
                        else if (j == QUEEN)
                            phase += engine::config::eval::queenPhase;
                    }
                }
            }
        }
        if (phase > engine::config::eval::totalPhase)
        {
            phase = engine::config::eval::totalPhase;
        }
    }

    inline void increment(const Move &m, Color us)
    {
        const Piece from_piece = m.get_from_piece();
        const Piece to_piece = m.get_to_piece();
        const int from_sq = m.get_from_sq();
        const int to_sq = m.get_to_sq();
        const Color them = (Color)!us;

        // 1. DÉPART DE LA PIÈCE
        remove_piece(from_piece, from_sq, us);

        // 2. CAPTURES (Classique ou En Passant)
        if (to_piece != NO_PIECE)
        {
            remove_piece(to_piece, to_sq, them);
            update_phase_on_capture(to_piece);
        }
        else if (m.get_flags() == Move::Flags::EN_PASSANT_CAP)
        {
            const int cap_sq = (us == WHITE) ? to_sq - 8 : to_sq + 8;
            remove_piece(PAWN, cap_sq, them);
        }

        // 3. ARRIVÉE DE LA PIÈCE (Promotion incluse)
        Piece final_piece = m.is_promotion() ? m.get_promo_piece() : from_piece;
        add_piece(final_piece, to_sq, us);

        if (m.is_promotion())
            update_phase_on_promotion();

        // 4. ROQUES (Mouvement de la Tour)
        if (m.get_flags() == Move::Flags::KING_CASTLE)
        {
            int r_from = (us == WHITE) ? Square::h1 : Square::h8;
            int r_to = (us == WHITE) ? Square::f1 : Square::f8;
            remove_piece(ROOK, r_from, us);
            add_piece(ROOK, r_to, us);
        }
        else if (m.get_flags() == Move::Flags::QUEEN_CASTLE)
        {
            int r_from = (us == WHITE) ? Square::a1 : Square::a8;
            int r_to = (us == WHITE) ? Square::d1 : Square::d8;
            remove_piece(ROOK, r_from, us);
            add_piece(ROOK, r_to, us);
        }
    }

    inline void decrement(const Move &m, Color us)
    {
        const Piece from_piece = m.get_from_piece();
        const Piece to_piece = m.get_to_piece();
        const int from_sq = m.get_from_sq();
        const int to_sq = m.get_to_sq();
        const Color them = (Color)!us;

        // 1. ANNULER L'ARRIVÉE SUR LA CASE 'TO'
        // Si c'était une promotion, on retire la Dame, sinon on retire la pièce d'origine
        Piece final_piece = m.is_promotion() ? m.get_promo_piece() : from_piece;
        remove_piece(final_piece, to_sq, us);

        if (m.is_promotion())
            restore_phase_on_promotion();

        // 2. ANNULER LE MOUVEMENT DE LA TOUR (SI ROQUE)
        if (m.get_flags() == Move::Flags::KING_CASTLE)
        {
            int r_from = (us == WHITE) ? Square::h1 : Square::h8;
            int r_to = (us == WHITE) ? Square::f1 : Square::f8;
            add_piece(ROOK, r_from, us);  // On remet la tour au coin
            remove_piece(ROOK, r_to, us); // On l'enlève de f1/f8
        }
        else if (m.get_flags() == Move::Flags::QUEEN_CASTLE)
        {
            int r_from = (us == WHITE) ? Square::a1 : Square::a8;
            int r_to = (us == WHITE) ? Square::d1 : Square::d8;
            add_piece(ROOK, r_from, us);  // On remet la tour au coin
            remove_piece(ROOK, r_to, us); // On l'enlève de d1/d8
        }

        // 3. RESTAURER LA PIÈCE CAPTURÉE
        if (to_piece != NO_PIECE)
        {
            add_piece(to_piece, to_sq, them);
            restore_phase_on_capture(to_piece);
        }
        else if (m.get_flags() == Move::Flags::EN_PASSANT_CAP)
        {
            const int cap_sq = (us == WHITE) ? to_sq - 8 : to_sq + 8;
            add_piece(PAWN, cap_sq, them);
        }

        // 4. RETOUR À LA CASE DE DÉPART
        add_piece(from_piece, from_sq, us);

        // On remet la position du Roi à jour (case d'origine)
        if (from_piece == KING)
            king_sq[us] = from_sq;
    }

private:
    // Dans EvalState : s'assurer que add_piece et remove_piece gèrent king_sq proprement
    inline void add_piece(Piece p, int sq, Color c)
    {
        int mirror = (c == WHITE) ? sq : sq ^ 56;

        // Utilisation des tables respectives MG et EG
        int pst_mg = engine::config::eval::mg_tables[p][mirror];
        int pst_eg = engine::config::eval::eg_tables[p][mirror];

        mg_pst[c] += pst_mg;
        eg_pst[c] += pst_eg;
        pieces_val[c] += engine::config::eval::pieces_score[p];

        if (p == PAWN)
            pawn_key ^= zobrist_table[PAWN + (c == BLACK ? 6 : 0)][sq];
        else if (p == KING)
            king_sq[c] = sq;
    }

    inline void remove_piece(Piece p, int sq, Color c)
    {
        int mirror = (c == WHITE) ? sq : sq ^ 56;

        int pst_mg = engine::config::eval::mg_tables[p][mirror];
        int pst_eg = engine::config::eval::eg_tables[p][mirror];

        mg_pst[c] -= pst_mg;
        eg_pst[c] -= pst_eg;
        pieces_val[c] -= engine::config::eval::pieces_score[p];

        if (p == PAWN)
            pawn_key ^= zobrist_table[PAWN + (c == BLACK ? 6 : 0)][sq];
    }
    inline void update_phase_on_capture(Piece p)
    {
        switch (p)
        {
        case KNIGHT:
            phase -= engine::config::eval::knightPhase;
            break;
        case BISHOP:
            phase -= engine::config::eval::bishopPhase;
            break;
        case ROOK:
            phase -= engine::config::eval::rookPhase;
            break;
        case QUEEN:
            phase -= engine::config::eval::queenPhase;
            break;
        default:
            break;
        }
        if (phase < 0)
            phase = 0;
    }
    inline void update_phase_on_promotion()
    {
        phase += engine::config::eval::queenPhase;
        if (phase > engine::config::eval::totalPhase)
            phase = engine::config::eval::totalPhase;
    }

    inline void restore_phase_on_capture(Piece p)
    {
        if (p == KNIGHT)
            phase += engine::config::eval::knightPhase;
        else if (p == BISHOP)
            phase += engine::config::eval::bishopPhase;
        else if (p == ROOK)
            phase += engine::config::eval::rookPhase;
        else if (p == QUEEN)
            phase += engine::config::eval::queenPhase;

        if (phase > engine::config::eval::totalPhase)
            phase = engine::config::eval::totalPhase;
    }

    inline void restore_phase_on_promotion()
    {
        // On retire la phase de la Dame qui a été annulée
        phase -= engine::config::eval::queenPhase;
        if (phase < 0)
            phase = 0;
    }
};
