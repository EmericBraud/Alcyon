#pragma once

// Mise a jour des features Full_Threats par deltas directs.
//
// L'ancienne methode (full_threats_incremental.hpp) collecte toutes les
// menaces autour des cases touchees AVANT le coup, puis APRES, et en fait la
// difference (NnueEval::filter_threats_diff) : deux collectes completes, dont
// la plupart des features s'annulent.
//
// Ici, le coup est decompose en changements elementaires (une piece retiree
// d'une case, une piece posee sur une case), appliques un par un a un petit
// etat local (occupation, bitboards, mailbox). Pour chaque changement sur la
// case s, les features qui changent sont exactement :
//   (a) celles ou la piece en s attaque (ses cibles, poussee de pion comprise) ;
//   (b) celles ou la piece en s est attaquee (attaquants de s, poussee inverse
//       de pion comprise) ;
//   (c) les rayons decouverts : chaque fou/tour/dame qui voit s voit, s vide,
//       la premiere piece au-dela -- cette menace apparait quand s se vide et
//       disparait quand s se remplit.
// Rien d'autre ne depend de l'occupation de s : pions, cavaliers et rois ont
// des cibles fixes, et une cible n'existe que si elle est occupee. Chaque
// changement donne donc la difference exacte d'ensembles entre deux etats
// consecutifs, et leur somme est la difference avant/apres du coup.
//
// Une feature retiree puis rajoutee dans le meme coup (ou l'inverse) est
// annulee au moment de l'emission, comme le faisait filter_threats_diff ; de
// toute facon l'accumulateur (sommes int16 modulo 2^16) donnerait le meme
// resultat.

#include <cstdint>

#include "common/cpu.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"
#include "core/piece/color.hpp"
#include "core/piece/piece.hpp"
#include "engine/eval/nnue/fixed_int_list.hpp"
#include "engine/eval/nnue/full_threats_encoder.hpp"

namespace nnue::threats
{
    template <int Cap>
    class ThreatDelta
    {
        // Meme codage que Board::mailbox : (couleur << 3) | type, case vide =
        // (NO_COLOR << 3) | NO_PIECE.
        static constexpr std::uint8_t Empty = (NO_COLOR << 3) | NO_PIECE;

        U64 occ;
        std::array<U64, constants::NumPieceVariants> pcs;
        std::uint8_t mb[64];

        const int ksq[2];
        const bool on[2];
        FixedIntList<Cap> *rem[2];
        FixedIntList<Cap> *add[2];
        // debut de la partie menaces de chaque liste (avant : bascules HalfKA)
        int rem0[2], add0[2];

        Piece type_at(int sq) const { return static_cast<Piece>(mb[sq] & 7); }
        Color color_at(int sq) const { return static_cast<Color>((mb[sq] >> 3) & 3); }
        U64 bb(Piece p, Color c) const { return pcs[get_piece_index(p, c)]; }

        // Ajoute idx a `to`, sauf s'il est deja dans `from` (partie menaces) :
        // alors les deux s'annulent.
        static void push_cancel(FixedIntList<Cap> &to, FixedIntList<Cap> &from, int from0, int idx)
        {
            const int n = from.size();
            for (int i = from0; i < n; ++i)
                if (from[i] == idx)
                {
                    from[i] = from[n - 1];
                    from.pop_back();
                    return;
                }
            to.push_back(idx);
        }

        template <Color P>
        void emit_one(Piece at, Color ac, int from, int to, Piece dt, Color dc, bool adding)
        {
            const int idx = threat_index<P>(at, ac, from, to, dt, dc, ksq[P]);
            if (idx < 0)
                return;
            if (adding)
                push_cancel(*add[P], *rem[P], rem0[P], idx);
            else
                push_cancel(*rem[P], *add[P], add0[P], idx);
        }

        void emit(Piece at, Color ac, int from, int to, Piece dt, Color dc, bool adding)
        {
            if (on[WHITE])
                emit_one<WHITE>(at, ac, from, to, dt, dc, adding);
            if (on[BLACK])
                emit_one<BLACK>(at, ac, from, to, dt, dc, adding);
        }

        // Cibles (occupees) de la piece (t, c) en sq : meme regle que
        // piece_attack_targets (full_threats_incremental.hpp).
        U64 targets(Piece t, Color c, int sq) const
        {
            switch (t)
            {
            case PAWN:
            {
                U64 a = pawn_attacks_no_blockers(c, sq);
                const int push = sq + (c == WHITE ? 8 : -8);
                if (push >= 0 && push < 64)
                    a |= 1ULL << push;
                return a & occ;
            }
            case KNIGHT:
                return MoveGen::KnightAttacks[sq] & occ;
            case BISHOP:
                return MoveGen::generate_bishop_moves(sq, occ) & occ;
            case ROOK:
                return MoveGen::generate_rook_moves(sq, occ) & occ;
            case QUEEN:
                return (MoveGen::generate_rook_moves(sq, occ) | MoveGen::generate_bishop_moves(sq, occ)) & occ;
            default:
                return 0ULL; // roi : jamais attaquant
            }
        }

        // Changement elementaire : piece (t, c) posee en sq (adding) ou
        // retiree de sq. Les features sont calculees dans l'etat ou la piece
        // est presente.
        void toggle(int sq, Piece t, Color c, bool adding)
        {
            const U64 bit = 1ULL << sq;
            if (adding)
            {
                occ |= bit;
                pcs[get_piece_index(t, c)] |= bit;
                mb[sq] = static_cast<std::uint8_t>((c << 3) | t);
            }

            // (a) la piece en sq attaque
            U64 tg = targets(t, c, sq);
            while (tg)
            {
                const int to = cpu::pop_lsb(tg);
                emit(t, c, sq, to, type_at(to), color_at(to), adding);
            }

            // (b) la piece en sq est attaquee
            const U64 bq = bb(BISHOP, WHITE) | bb(BISHOP, BLACK) | bb(QUEEN, WHITE) | bb(QUEEN, BLACK);
            const U64 rq = bb(ROOK, WHITE) | bb(ROOK, BLACK) | bb(QUEEN, WHITE) | bb(QUEEN, BLACK);
            const U64 diag = MoveGen::generate_bishop_moves(sq, occ);
            const U64 orth = MoveGen::generate_rook_moves(sq, occ);
            const U64 dsl = diag & bq;
            const U64 osl = orth & rq;
            U64 att = (MoveGen::PawnAttacksBlack[sq] & bb(PAWN, WHITE)) |
                      (MoveGen::PawnAttacksWhite[sq] & bb(PAWN, BLACK)) |
                      (MoveGen::KnightAttacks[sq] & (bb(KNIGHT, WHITE) | bb(KNIGHT, BLACK))) |
                      dsl | osl;
            // poussee inverse : pion blanc en sq-8, pion noir en sq+8
            if (sq >= 8 && (bb(PAWN, WHITE) & (bit >> 8)))
                att |= bit >> 8;
            if (sq < 56 && (bb(PAWN, BLACK) & (bit << 8)))
                att |= bit << 8;
            while (att)
            {
                const int from = cpu::pop_lsb(att);
                emit(type_at(from), color_at(from), from, sq, t, c, adding);
            }

            // (c) rayons decouverts : sq vide, chaque slider qui voyait sq
            // voit la premiere piece au-dela.
            const U64 occ_wo = occ & ~bit;
            U64 s = dsl;
            while (s)
            {
                const int from = cpu::pop_lsb(s);
                U64 beyond = MoveGen::generate_bishop_moves(from, occ_wo) & ~MoveGen::generate_bishop_moves(from, occ) & occ_wo;
                if (beyond)
                {
                    const int to = cpu::get_lsb_index(beyond);
                    emit(type_at(from), color_at(from), from, to, type_at(to), color_at(to), !adding);
                }
            }
            s = osl;
            while (s)
            {
                const int from = cpu::pop_lsb(s);
                U64 beyond = MoveGen::generate_rook_moves(from, occ_wo) & ~MoveGen::generate_rook_moves(from, occ) & occ_wo;
                if (beyond)
                {
                    const int to = cpu::get_lsb_index(beyond);
                    emit(type_at(from), color_at(from), from, to, type_at(to), color_at(to), !adding);
                }
            }

            if (!adding)
            {
                occ &= ~bit;
                pcs[get_piece_index(t, c)] &= ~bit;
                mb[sq] = Empty;
            }
        }

    public:
        // Etat initial = position avant le coup. Perspectives actives : `w`,
        // `b` (pour un coup de roi, seule celle de l'adversaire est
        // incrementale).
        ThreatDelta(const Board &board, bool w, bool b,
                    FixedIntList<Cap> &rem_w, FixedIntList<Cap> &add_w,
                    FixedIntList<Cap> &rem_b, FixedIntList<Cap> &add_b)
            : occ(board.occupancies[NO_COLOR]), pcs(board.pieces_occ),
              ksq{board.king_sq[WHITE], board.king_sq[BLACK]}, on{w, b},
              rem{&rem_w, &rem_b}, add{&add_w, &add_b},
              rem0{rem_w.size(), rem_b.size()}, add0{add_w.size(), add_b.size()}
        {
            std::memcpy(mb, board.mailbox, 64);
        }

        void remove_piece(int sq, Piece t, Color c) { toggle(sq, t, c, false); }
        void add_piece(int sq, Piece t, Color c) { toggle(sq, t, c, true); }
    };
}
