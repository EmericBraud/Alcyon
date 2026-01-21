#include "worker.hpp"
#include "engine/engine_manager.hpp"

template <Color Us>
int SearchWorker::qsearch(int alpha, int beta, int ply)
{
    if (check_stop())
        return alpha;

    // 2. Sondage de la Transposition Table (TT)
    // Utilisation du ply pour normaliser les scores de mat récupérés
    int tt_score;
    TTFlag flag;
    Move tt_move = 0;
    if (shared_tt.probe(board.get_hash(), 0, ply, alpha, beta, tt_score, tt_move, flag))
        return tt_score;

    bool in_check = board.is_king_attacked<Us>();
    int stand_pat = -engine::config::eval::Inf;

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
        if (m == tt_move) // On priorise le coup TT s'il existe
            list.scores[i] = 2000000;
        else if (in_check)
            list.scores[i] = score_move<Us>(m, tt_move, ply, 0);
        else
            list.scores[i] = score_capture(m); // Juste MVV/LVA, pas de SEE !
    }

    int best_score = in_check ? -engine::config::eval::Inf : stand_pat;
    int moves_searched = 0;
    int alpha_orig = alpha;
    Move best_move = 0;

    // 6. Boucle de recherche
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);

        // --- DELTA PRUNING (Seulement hors échec) ---
        if (!in_check)
        {
            Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
            int victim_val = Eval::get_piece_score(target);
            int promo_bonus = (m.get_flags() & Move::PROMOTION_MASK) ? 800 : 0;
            if (stand_pat + victim_val + promo_bonus + 200 < alpha)
                continue;
            int attacker = m.get_from_piece();

            if (Eval::get_piece_score(attacker) > victim_val)
            {
                if (see<Us>(m.get_to_sq(), static_cast<Piece>(target), static_cast<Piece>(attacker), m.get_from_sq()) < 0)
                    continue;
            }
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

    if (moves_searched == 0)
    {
        if (in_check)
        {
            int score = -engine::config::eval::MateScore + ply;
            shared_tt.store(board.get_hash(), 0, ply, score, TT_EXACT, 0);
            return score;
        }
    }

    // 8. Sauvegarde TT finale
    flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), 0, ply, best_score, flag, best_move);

    return best_score;
}
// Dans SearchWorker (ou inline)
inline int SearchWorker::score_capture(const Move &move) const
{
    // Utilisation directe de la table pour éviter les calculs
    // On suppose que MvvLvaTable est accessible (namespace config ou membre)
    // format: MvvLvaTable[victim][attacker]

    int score = 0;

    if (move.get_flags() == Move::EN_PASSANT_CAP)
    {
        // Pion mange Pion en passant
        score = engine::config::eval::MvvLvaTable[PAWN][PAWN];
    }
    else
    {
        score = engine::config::eval::MvvLvaTable[move.get_to_piece()][move.get_from_piece()];
    }

    // Offset pour que les captures soient triées AVANT les coups calmes (killers, etc)
    score += 1000000;

    if (move.is_promotion())
    {
        // Bonus simple pour promotion (souvent Dame)
        score += 10000;
    }

    return score;
}

template int SearchWorker::qsearch<WHITE>(int alpha, int beta, int ply);
template int SearchWorker::qsearch<BLACK>(int alpha, int beta, int ply);