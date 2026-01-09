#include "engine/engine.hpp"
#include "engine/engine_manager.hpp"

template <Color Us>
int SearchWorker::qsearch(int alpha, int beta, int ply)
{
    // 1. Vérification de l'arrêt et mise à jour des nœuds (Optimisé)
    if ((local_nodes & 32767) == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;

        if (thread_id == 0 && manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
        }
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;
    }
    ++local_nodes;

    // 2. Sondage de la Transposition Table (TT)
    // Utilisation du ply pour normaliser les scores de mat récupérés
    int tt_score;
    Move tt_move = 0;
    if (shared_tt.probe(board.get_hash(), 0, ply, alpha, beta, tt_score, tt_move))
        return tt_score;

    bool in_check = board.is_king_attacked<Us>();
    int stand_pat = -INF;

    // 3. Standing Pat (Évaluation statique)
    // On ne l'utilise que si on n'est pas en échec, car une position en échec est instable
    if (!in_check)
    {
        stand_pat = Eval::eval_relative<Us>(board, alpha, beta);
        if (stand_pat >= beta)
            return beta;
        if (stand_pat > alpha)
            alpha = stand_pat;
    }

    // 4. Génération des coups
    MoveList list;
    if (in_check)
    {
        // Si on est en échec, on doit générer TOUTES les évasions (pas seulement les captures)
        // pour éviter d'être aveugle aux mats forcés.
        MoveGen::generate_pseudo_legal_moves<Us>(board, list);
    }
    else
    {
        MoveGen::generate_pseudo_legal_captures<Us>(board, list);
    }

    // 5. Tri des coups (SEE + MVV-LVA)
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list[i];
        if (in_check)
        {
            // Pour les évasions, on peut réutiliser ton score_move habituel ou MVV-LVA simple
            list.scores[i] = score_move(m, board, tt_move, ply, 0);
        }
        else
        {
            Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
            int see_val = see(m.get_to_sq(), target, m.get_from_piece(), Us, m.get_from_sq());

            if (see_val < 0)
                list.scores[i] = -1000000; // Capture perdante
            else
                list.scores[i] = 1000000 + see_val + score_capture(m);
        }
    }

    int best_score = in_check ? -INF : stand_pat;
    int moves_searched = 0;
    int alpha_orig = alpha;
    Move best_move = 0;

    // 6. Boucle de recherche
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);

        // Élagage SEE : On ignore les captures perdantes sauf si on doit absolument sortir d'un échec
        if (!in_check && list.scores[i] < 0)
            continue;

        // --- DELTA PRUNING (Seulement hors échec) ---
        if (!in_check)
        {
            Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
            int victim_val = Eval::get_piece_score(target);
            int promo_bonus = (m.get_flags() & Move::PROMOTION_MASK) ? 800 : 0;
            if (stand_pat + victim_val + promo_bonus + 200 < alpha)
                continue;
        }

        board.play<Us>(m);
        if (board.is_king_attacked<Us>())
        {
            board.unplay<Us>(m);
            continue;
        }

        moves_searched++;
        // Appel récursif avec ply+1 pour la détection précise des mats
        int score = -qsearch<!Us>(-beta, -alpha, ply + 1);
        board.unplay<Us>(m);

        if (score >= beta)
        {
            // Stockage avec normalisation du score de mat (via ply interne à store)
            shared_tt.store(board.get_hash(), 0, ply, beta, TT_BETA, m);
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

    // 7. Détection de Mat (Si aucun coup n'a pu être joué alors qu'on est en échec)
    if (in_check && moves_searched == 0)
        return -MATE_SCORE + ply;

    // 8. Sauvegarde TT finale
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), 0, ply, best_score, flag, best_move);

    return best_score;
}
int SearchWorker::score_capture(const Move &move) const
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
        score += Eval::get_piece_score(move.get_promo_piece());
    }

    return score;
}

template int SearchWorker::qsearch<WHITE>(int alpha, int beta, int ply);
template int SearchWorker::qsearch<BLACK>(int alpha, int beta, int ply);