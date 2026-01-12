#include "engine/eval/pos_eval.hpp"

PawnTable pawn_table;

namespace Eval
{
    static const int *mobility_bonus_tables[] = {
        nullptr, // PAWN (pas géré ici)
        knight_mob,
        bishop_mob,
        rook_mob,
        queen_mob};
}

// Transforme un bitboard de pions en un masque où chaque bit
// représente une colonne (file) occupée (8 bits utilisés).
inline uint8_t get_pawn_files(bitboard pawns)
{
    // On "écrase" les pions verticalement pour ne garder que les colonnes
    pawns |= (pawns >> 32);
    pawns |= (pawns >> 16);
    pawns |= (pawns >> 8);
    return static_cast<uint8_t>(pawns & 0xFF);
}
int Eval::evaluate_castling_and_safety(Color us, const Board &board)
{
    const Color them = (Color)!us;
    const int king_sq = board.get_eval_state().king_sq[us];
    const int king_file = king_sq & 7;

    const bitboard our_pawns = board.get_piece_bitboard(us, PAWN);
    const bitboard enemy_pawns = board.get_piece_bitboard(them, PAWN);
    const bitboard enemy_heavies = board.get_piece_bitboard(them, ROOK) | board.get_piece_bitboard(them, QUEEN);

    // 1. Compression des fichiers (8-bit masks)
    uint8_t our_files = get_pawn_files(our_pawns);
    uint8_t enemy_files = get_pawn_files(enemy_pawns);
    uint8_t vicinity = masks::king_vicinity_files[king_file];

    // 2. Détection logique des colonnes (Branchless)
    uint8_t open = vicinity & ~our_files & ~enemy_files;
    uint8_t semi_open = vicinity & ~our_files & enemy_files;

    // 3. Calcul du score de base par popcount (Instruction matérielle ultra-rapide)
    int score = std::popcount(static_cast<uint8_t>(open)) * -35;
    score += std::popcount(static_cast<uint8_t>(semi_open)) * -15;

    // 4. Masques 64-bit : On traite uniquement les 3 colonnes du voisinage
    bitboard open_files_bb = 0;
    bitboard semi_files_bb = 0;

    // On utilise les bornes fixes pour forcer l'unrolling du compilateur
    // On itère sur les 3 colonnes potentielles autour du roi
    for (int f = king_file - 1; f <= king_file + 1; ++f)
    {
        // On ignore les colonnes hors échiquier sans if (branchless mask)
        // (f >= 0 && f <= 7) est transformé en masque 0 ou -1
        int64_t valid_file_mask = -static_cast<int64_t>(f >= 0 && f <= 7);
        int clamped_f = f & 7; // Sécurité d'indexation

        // Extraction du bit correspondant à la colonne f dans nos masques 8-bits
        // Si le bit est à 1, on applique le masque de colonne 64-bit
        open_files_bb |= (masks::file[clamped_f] & -static_cast<int64_t>((open >> clamped_f) & 1)) & valid_file_mask;
        semi_files_bb |= (masks::file[clamped_f] & -static_cast<int64_t>((semi_open >> clamped_f) & 1)) & valid_file_mask;
    }

    // 5. Pénalité si des pièces lourdes adverses occupent ces couloirs
    score += std::popcount(enemy_heavies & open_files_bb) * -45;
    score += std::popcount(enemy_heavies & semi_files_bb) * -25;

    return score;
}
void Eval::evaluate_pawns(Color color, const Board &board, int &mg, int &eg)
{
    const bitboard our_pawns = board.get_piece_bitboard(color, PAWN);
    const bitboard enemy_pawns = board.get_piece_bitboard(!color, PAWN);

    // 1. DETECTION GLOBALE DES PIONS DOUBLÉS (Zéro boucle)
    // On identifie les pions qui ont un autre pion allié "sous eux" sur la même colonne
    bitboard doubled_mask = our_pawns & ((our_pawns >> 8) | (our_pawns >> 16) | (our_pawns >> 24) |
                                         (our_pawns >> 32) | (our_pawns >> 40) | (our_pawns >> 48) | (our_pawns >> 56));
    // On compte le nombre de colonnes contenant des pions doublés
    int num_doubled_files = std::popcount(get_pawn_files(doubled_mask));
    mg -= num_doubled_files * 15;
    eg -= num_doubled_files * 20;

    // 2. DETECTION GLOBALE DES PIONS ISOLÉS (Zéro boucle)
    uint8_t f = get_pawn_files(our_pawns);
    // Une colonne est isolée si ses voisines (f << 1) et (f >> 1) sont vides
    uint8_t iso_files = f & ~((f << 1) | (f >> 1));
    int num_iso = std::popcount(iso_files);
    mg -= num_iso * 20;
    eg -= num_iso * 25;

    // 3. BOUCLE ALLÉGÉE (Uniquement pour les pions passés)
    bitboard temp_pawns = our_pawns;
    while (temp_pawns)
    {
        const int sq = pop_lsb(temp_pawns);

        // Un pion est passé s'il n'y a aucun pion adverse devant lui (colonne + adjacentes)
        if (!(enemy_pawns & masks.passed[color][sq]))
        {
            const int rank = sq >> 3;
            const int relative_rank = (color == WHITE) ? rank : (7 - rank);

            static constexpr int passed_bonus_mg[] = {0, 5, 10, 20, 40, 70, 120, 0};
            static constexpr int passed_bonus_eg[] = {0, 10, 20, 40, 80, 150, 250, 0};

            mg += passed_bonus_mg[relative_rank];
            eg += passed_bonus_eg[relative_rank];
        }
    }
}
int Eval::eval(const Board &board, int alpha, int beta)
{
    const EvalState &state = board.get_eval_state();

    // 1. Matériel + PST (Incrémental : GRATUIT)
    // On fusionne material_score directement ici pour éviter une addition finale
    int mg_score = (state.mg_pst[WHITE] + state.pieces_val[WHITE]) -
                   (state.mg_pst[BLACK] + state.pieces_val[BLACK]);
    int eg_score = (state.eg_pst[WHITE] + state.pieces_val[WHITE]) -
                   (state.eg_pst[BLACK] + state.pieces_val[BLACK]);

    // 2. Structure des Pions (Cache Pawn Table)
    int mg_pawn = 0, eg_pawn = 0;

    if (!pawn_table.probe(state.pawn_key, mg_pawn, eg_pawn))
    {
        int mg_w = 0, eg_w = 0, mg_b = 0, eg_b = 0;
        evaluate_pawns(WHITE, board, mg_w, eg_w);
        evaluate_pawns(BLACK, board, mg_b, eg_b);
        mg_pawn = mg_w - mg_b;
        eg_pawn = eg_w - eg_b;
        pawn_table.store(state.pawn_key, mg_pawn, eg_pawn);
    }
    mg_score += mg_pawn;
    eg_score += eg_pawn;
    int base_score = (mg_score * state.phase + eg_score * (totalPhase - state.phase)) / totalPhase;
    const int margin = 110;
    if (base_score >= beta + margin)
        return base_score;
    if (base_score <= alpha - margin)
        return base_score;
    // Cache des bitboards d'occupation pour éviter les appels répétés
    const U64 occ_all = board.get_occupancy(NO_COLOR);
    const U64 occ_white = board.get_occupancy(WHITE);
    const U64 occ_black = board.get_occupancy(BLACK);

    // 3. Mobilité et Sécurité
    for (int color = WHITE; color <= BLACK; ++color)
    {
        int bonus = 0;
        const Color us = static_cast<Color>(color);
        const U64 our_occ = (us == WHITE) ? occ_white : occ_black;

        // Sécurité du Roi
        bonus += evaluate_castling_and_safety(us, board);

        if (std::popcount(board.get_piece_bitboard(us, BISHOP)) >= 2)
        {
            bonus += 30; // Bonus milieu de jeu
            bonus += 50; // Bonus fin de partie (plus important car le plateau est ouvert)
        }

        // Mobilité optimisée
        for (int piece = KNIGHT; piece <= QUEEN; ++piece)
        {
            bitboard bb = board.get_piece_bitboard(us, piece);
            const int *bonus_table = mobility_bonus_tables[piece];

            while (bb)
            {
                const int sq = pop_lsb(bb);

                // On utilise les fonctions spécifiques de Magic Bitboards si possible
                // au lieu du get_pseudo_moves_mask générique qui est plus lent.
                U64 moves;
                if (piece == KNIGHT)
                    moves = MoveGen::KnightAttacks[sq];
                else if (piece == BISHOP)
                    moves = MoveGen::generate_bishop_moves(sq, occ_all);
                else if (piece == ROOK)
                    moves = MoveGen::generate_rook_moves(sq, occ_all);
                else
                    moves = MoveGen::generate_bishop_moves(sq, occ_all) | MoveGen::generate_rook_moves(sq, occ_all);

                // On ne compte que les cases libres ou occupées par l'adversaire
                const int count = std::popcount(moves & ~our_occ);

                // Accès direct à la table (Branchless)
                // Pour la dame, on sature à 27 via std::min
                bonus += bonus_table[(piece == QUEEN) ? std::min(count, 27) : count];
            }
        }

        // Application du bonus (Branchless via multiplication par le signe)
        const int sign = (us == WHITE) ? 1 : -1;
        mg_score += bonus * sign;
        eg_score += bonus * sign;
    }

    if (std::abs(eg_score) > 200)
    {
        // On identifie qui gagne
        const Color winner = (eg_score > 0) ? WHITE : BLACK;
        const Color loser = (Color)!winner;

        const int wk = state.king_sq[WHITE];
        const int bk = state.king_sq[BLACK];

        // 1. Bonus pour pousser le roi adverse vers le bord (Center Manhatten distance)
        const int loser_king = state.king_sq[loser];
        int k_file = loser_king & 7;
        int k_rank = loser_king >> 3;

        // Distance du centre (0 à 3)
        int dist_from_center = std::max(std::abs(k_file - 3), std::abs(k_rank - 3));

        // 2. Bonus pour approcher notre Roi du Roi adverse
        int dist_between_kings = king_distance(wk, bk);

        // On combine les deux :
        // + on pousse au bord, + on gagne. + on est proche, + on gagne.
        int mop_up = (dist_from_center * 10) + (14 - dist_between_kings) * 5;

        // On applique le bonus selon le gagnant
        if (winner == WHITE)
            eg_score += mop_up;
        else
            eg_score -= mop_up;
    }

    // 4. Interpolation finale (material_score a été fusionné à l'étape 1)
    return (mg_score * state.phase + eg_score * (totalPhase - state.phase)) / totalPhase;
}
void Eval::print_pawn_stats()
{
    logs::debug << "Pawn Table Stats:" << std::endl;
    logs::debug << " - Hits:   " << pawn_table.hits << std::endl;
    logs::debug << " - Misses: " << pawn_table.misses << std::endl;
    logs::debug << " - Rate:   " << pawn_table.get_hit_rate() << "%" << std::endl;
}