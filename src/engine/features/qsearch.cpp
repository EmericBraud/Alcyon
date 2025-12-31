#include "engine/engine.hpp"
int SearchWorker::qsearch(int alpha, int beta)
{
    if (((++local_nodes) & 2047) == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;
        if (thread_id == 0)
        {
            auto now = Clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_ref).count();
            if (elapsed >= time_limit_ms_ref)
            {
                shared_stop.store(true, std::memory_order_relaxed);
            }
        }
        // Si l'arrêt est demandé (par temps ou par un autre thread), on quitte
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;
    }

    // 2. Sondage de la Transposition Table (TT)
    int tt_score;
    Move tt_move = 0;
    // En qsearch, la profondeur est considérée comme 0
    bool tt_hit = shared_tt.probe(board.get_hash(), 0, 0, alpha, beta, tt_score, tt_move);
    if (tt_hit)
        return tt_score;

    // 3. Standing Pat (Évaluation statique)
    int stand_pat = Eval::eval_relative(board.get_side_to_move(), board, alpha, beta);
    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    const Color player = board.get_side_to_move();

    // 4. Génération des captures pseudo-légales
    MoveList list;
    MoveGen::generate_pseudo_legal_captures(board, player, list);

    // 5. TRI HYBRIDE : SEE + MVV-LVA
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list[i];
        Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();

        // Calcul du SEE pour filtrer et trier
        int see_val = see(m.get_to_sq(), target, m.get_from_piece(), player, m.get_from_sq());

        if (see_val < 0)
            list.scores[i] = -1000000; // Mauvaise capture
        else
            list.scores[i] = 1000000 + see_val + score_capture(m);
    }

    int best_score = stand_pat;
    int alpha_orig = alpha;
    Move best_move = 0;

    // 6. Boucle de recherche
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);

        // Élagage SEE
        if (list.scores[i] < 0)
            continue;

        // --- DELTA PRUNING ---
        Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
        int victim_val = Eval::get_piece_score(target);
        int promo_bonus = (m.get_flags() & Move::PROMOTION_MASK) ? 800 : 0;

        if (stand_pat + victim_val + promo_bonus + 200 < alpha)
            continue;

        // --- EXÉCUTION ---
        shared_tt.prefetch(board.get_hash_after(m));
        board.play(m);

        if (board.is_king_attacked(player))
        {
            board.unplay(m);
            continue;
        }

        int score = -qsearch(-beta, -alpha);
        board.unplay(m);

        // Sortie immédiate si le temps est écoulé pendant la récursion
        if (shared_stop.load(std::memory_order_relaxed))
            return alpha;

        // --- MISE À JOUR ALPHA-BETA ---
        if (score >= beta)
        {
            shared_tt.store(board.get_hash(), 0, 0, beta, TT_BETA, m);
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

    // 7. Sauvegarde TT
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), 0, 0, best_score, flag, best_move);

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
        score += Eval::get_piece_score(QUEEN);
    }

    return score;
}
