#pragma once
#include "engine/zobrist.hpp"

static constexpr int mg_pawn_table[64] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50, // Rang 7 : Menace de promotion
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 5, 10, 10, 5, 5, 5, // Rang 3 : Encourager l'avance légère
    0, 0, 0, 0, 0, 0, 0, 0,   // Rang 2 : Neutre (on ne punit plus l'avance)
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

static const int pawnPhase = 0;
static const int knightPhase = 1;
static const int bishopPhase = 1;
static const int rookPhase = 2;
static const int queenPhase = 4;
static const int totalPhase = 24;

static const int phase_values[] = {pawnPhase, knightPhase, bishopPhase, rookPhase, queenPhase, totalPhase};

struct EvalState
{
    int16_t mg_pst[2];  // Score Middle-Game PST + Matériel par couleur
    int16_t eg_pst[2];  // Score End-Game PST + Matériel par couleur
    int8_t phase;       // Phase actuelle (0 à 24)
    U64 pawn_key;       // Clé Zobrist spécifique aux pions (pour la Pawn Table)
    uint8_t king_sq[2]; // Position des rois
    int16_t pieces_val[2];

    EvalState() = default;
    EvalState(const std::array<bitboard, N_PIECES_TYPE> occupancy)
    {
        mg_pst[0] = mg_pst[1] = eg_pst[0] = eg_pst[1] = 0;
        pieces_val[0] = pieces_val[1] = 0;
        phase = 0;
        pawn_key = 0;

        for (int i = WHITE; i <= BLACK; ++i)
        {
            for (int j = PAWN; j <= KING; ++j)
            {
                U64 mask = occupancy[j + i * N_PIECES_TYPE_HALF];
                while (mask)
                {
                    int sq = pop_lsb(mask);
                    add_piece(static_cast<Piece>(j), sq, static_cast<Color>(i));

                    if (j != PAWN && j != KING)
                    {
                        if (j == KNIGHT)
                            phase += knightPhase;
                        else if (j == BISHOP)
                            phase += bishopPhase;
                        else if (j == ROOK)
                            phase += rookPhase;
                        else if (j == QUEEN)
                            phase += queenPhase;
                    }
                }
            }
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
            // Correction : Utiliser le square de capture EP pour retirer le pion
            const int ep_sq = m.get_prev_en_passant(us);
            remove_piece(PAWN, ep_sq, them);
        }

        // 3. ARRIVÉE DE LA PIÈCE (Promotion incluse)
        Piece final_piece = m.is_promotion() ? QUEEN : from_piece;
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
        Piece final_piece = m.is_promotion() ? QUEEN : from_piece;
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
            const int ep_sq = m.get_prev_en_passant(us);
            add_piece(PAWN, ep_sq, them);
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
        int pst_mg = mg_tables[p][mirror];
        int pst_eg = (p == KING) ? eg_king_table[mirror] : pst_mg;

        mg_pst[c] += pst_mg;
        eg_pst[c] += pst_eg;
        pieces_val[c] += pieces_score[p];

        if (p == PAWN)
            pawn_key ^= zobrist_table[PAWN + (c == BLACK ? 6 : 0)][sq];
        else if (p == KING)
            king_sq[c] = sq; // Mise à jour systématique
    }

    inline void remove_piece(Piece p, int sq, Color c)
    {
        int mirror = (c == WHITE) ? sq : sq ^ 56;
        int pst_mg = mg_tables[p][mirror];
        int pst_eg = (p == KING) ? eg_king_table[mirror] : pst_mg;

        mg_pst[c] -= pst_mg;
        eg_pst[c] -= pst_eg;
        pieces_val[c] -= pieces_score[p];

        if (p == PAWN)
            pawn_key ^= zobrist_table[PAWN + (c == BLACK ? 6 : 0)][sq];
        // Note: On ne touche pas à king_sq ici, add_piece le fera au prochain mouvement
    }
    inline void update_phase_on_capture(Piece p)
    {
        switch (p)
        {
        case KNIGHT:
            phase -= knightPhase;
            break;
        case BISHOP:
            phase -= bishopPhase;
            break;
        case ROOK:
            phase -= rookPhase;
            break;
        case QUEEN:
            phase -= queenPhase;
            break;
        default:
            break;
        }
        if (phase < 0)
            phase = 0;
    }
    inline void update_phase_on_promotion()
    {
        phase += queenPhase;
        if (phase > totalPhase)
            phase = totalPhase;
    }

    inline void restore_phase_on_capture(Piece p)
    {
        if (p == KNIGHT)
            phase += knightPhase;
        else if (p == BISHOP)
            phase += bishopPhase;
        else if (p == ROOK)
            phase += rookPhase;
        else if (p == QUEEN)
            phase += queenPhase;

        if (phase > totalPhase)
            phase = totalPhase;
    }

    inline void restore_phase_on_promotion()
    {
        // On retire la phase de la Dame qui a été annulée
        phase -= 4;
        if (phase < 0)
            phase = 0;
    }
};
