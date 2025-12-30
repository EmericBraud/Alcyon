#include "engine/engine.hpp"
int Engine::qsearch(int alpha, int beta)
{
    // 1. Contrôles de sécurité (Temps et Nœuds)
    if ((total_nodes & 2047) == 0)
        check_time();
    if (time_up)
        return Eval::eval_relative(board.get_side_to_move(), board, alpha, beta);

    q_nodes++;
    total_nodes++;

    // 2. Sondage de la Transposition Table (TT)
    int tt_score;
    Move tt_move = 0;
    // En qsearch, on utilise souvent depth = 0 ou une constante pour la TT
    bool tt_hit = tt.probe(board.get_hash(), 0, 0, alpha, beta, tt_score, tt_move);
    if (tt_hit)
        return tt_score;

    // 3. Standing Pat (Évaluation statique)
    // On considère que ne rien faire est une option (on peut "pat" si la position est bonne)
    int stand_pat = Eval::eval_relative(board.get_side_to_move(), board, alpha, beta);
    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    const Color player = board.get_side_to_move();

    // 4. Génération des captures pseudo-légales
    MoveList list;
    MoveGen::generate_pseudo_legal_captures(board, player, list);

    // 5. TRI HYBRIDE : Calcul des scores (SEE + MVV-LVA)
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list[i];
        Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();

        // On calcule la valeur réelle de l'échange via SEE
        int see_val = see(m.get_to_sq(), target, m.get_from_piece(), player, m.get_from_sq());

        if (see_val < 0)
        {
            // Mauvaises captures : on les marque pour les ignorer plus tard
            list.scores[i] = -1000000;
        }
        else
        {
            // Bonnes captures ou échanges égaux : SEE sert de base, MVV-LVA affine le tri
            list.scores[i] = 1000000 + see_val + score_capture(m);
        }
    }

    int best_score = stand_pat;
    int alpha_orig = alpha;
    Move best_move = 0;

    // 6. Boucle de recherche sur les captures
    for (int i = 0; i < list.count; ++i)
    {
        // On récupère le meilleur coup de la liste selon les scores calculés
        Move &m = list.pick_best_move(i);

        // --- FILTRAGE SEE ---
        // On ignore les captures qui perdent du matériel (SEE < 0)
        if (list.scores[i] < 0)
            continue;

        // --- DELTA PRUNING ---
        // Si même avec le gain de la capture + un bonus de promotion, on n'atteint pas alpha
        Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
        int victim_val = Eval::get_piece_score(target);
        int promo_bonus = (m.get_flags() & Move::PROMOTION_MASK) ? 800 : 0;

        if (stand_pat + victim_val + promo_bonus + 200 < alpha)
            continue;

        // --- EXECUTION DU COUP ---
        const U64 next_hash = board.get_hash_after(m);
        tt.prefetch(next_hash);

        board.play(m);

        // Vérification de la légalité (on n'a pas le droit de laisser son roi en échec)
        if (board.is_king_attacked(player))
        {
            _mm_prefetch((const char *)&(board.get_history())->back(), _MM_HINT_T0);
            board.unplay(m);
            continue;
        }

        // Appel récursif (Negamax)
        int score = -qsearch(-beta, -alpha);

        _mm_prefetch((const char *)&(board.get_history())->back(), _MM_HINT_T0);
        board.unplay(m);

        if (time_up)
            break;

        // --- MISE À JOUR ALPHA-BETA ---
        if (score >= beta)
        {
            tt.store(board.get_hash(), 0, 0, beta, TT_BETA, m);
            return beta;
        }

        if (score > best_score)
        {
            best_score = score;
            if (score > alpha)
            {
                alpha = score;
                best_move = m;
            }
        }
    }

    // 7. Sauvegarde dans la Table de Transposition
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    tt.store(board.get_hash(), 0, 0, best_score, flag, best_move);

    return best_score;
}
int Engine::score_capture(const Move &move) const
{
    int attacker = move.get_from_piece();

    // 2. Identifier la victime (Qui est mangé ?)
    int victim_val = 0;

    if (move.get_flags() == Move::EN_PASSANT_CAP)
    {
        victim_val = Eval::get_piece_score(PAWN);
    }
    else
    {
        // On regarde sur la case d'arrivée
        const int victim = move.get_to_piece();
        if (victim != NO_PIECE)
        {
            victim_val = Eval::get_piece_score(victim);
        }
    }

    // 3. Calcul du score
    int attacker_val = Eval::get_piece_score(attacker);

    // Formule MVV-LVA standard
    int score = 1000000 + (victim_val * 10 - attacker_val);

    // Bonus Promotion
    if (move.get_flags() == Move::PROMOTION_MASK)
    {
        score += Eval::get_piece_score(QUEEN);
    }

    return score;
}
