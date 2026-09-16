#include "worker.hpp"
#include "engine/engine_manager.hpp"

template <Color Us>
int SearchWorker::qsearch(int alpha, int beta, int ply)
{
    if (check_stop())
        return alpha;

    // Borne dure sur le ply, comme negamax en a une ("if (ply >= MaxDepth)").
    // La qsearch n'en avait pas : elle recurse en ply+1 et genere TOUTES les
    // evasions quand elle est en echec, donc une sequence d'echecs peut la
    // faire descendre arbitrairement loin. Deux consequences :
    //
    //  - score_move indexe killer_moves[ply], de taille [MaxDepth][2] : a
    //    ply >= 64 la lecture sortait du tableau, dans les tables d'history
    //    voisines de SearchWorker. Scores de coups aberrants, donc
    //    ordonnancement faux, silencieusement. Le meme danger etait deja
    //    connu et traite a UN seul endroit (safe_ply dans
    //    submit_current_position_to_gpu) -- ici c'est la cause racine.
    //  - chaque frame de qsearch porte une MoveList (~2 Ko), donc une
    //    recursion non bornee menace aussi la pile.
    //
    // Le garde est ici plutot qu'un clamp dans score_move : une comparaison
    // par noeud au lieu d'un clamp par coup, et il couvre les deux risques.
    //
    // Mesure : sur six positions dont une finale a echecs perpetuels, le
    // seldepth maximal observe est 26 et le garde ne se declenche JAMAIS.
    // C'est donc une assurance, pas une correction de comportement.
    if (ply >= engine_constants::search::MaxDepth)
        return Eval::lazy_eval_relative<Us>(board);

    // 2. Sondage de la Transposition Table (TT)
    // Utilisation du ply pour normaliser les scores de mat récupérés
    int tt_score;
    TTFlag flag;
    Move tt_move = 0;
    // La coupure qsearch est desactivee avec CHESS26_TT_NO_CUTOFF : sinon la
    // re-recherche sur TT pre-remplie gagnerait ici aussi par coupure, et
    // l'isolation de l'ordonnancement serait fausse. Le coup TT reste lu.
    if (shared_tt.probe(board.get_hash(), 0, ply, alpha, beta, tt_score, tt_move, flag) &&
        search::tt_cutoffs_enabled())
        return tt_score;

    bool in_check = board.is_king_attacked<Us>();
    int stand_pat = -engine_constants::eval::Inf;

    // 3. Standing Pat (Évaluation statique)
    // On ne l'utilise que si on n'est pas en échec, car une position en échec est instable
    if (!in_check)
    {
        // The GPU-eval queue precomputes CNN scores for the CHILDREN of PV
        // leaves (see gpu_queue.cpp), and those children are reached right
        // here, inside qsearch -- not in negamax. Consuming them anywhere
        // else measured at ~10 useful hits per multi-million-node search,
        // i.e. the subsystem was invisible to the search.
        //
        // Used as the stand-pat, which is where it belongs: what gets
        // stored is a quiesced value (the position is settled before it is
        // encoded), so it is the same KIND of quantity as the static eval
        // it replaces, only computed off-thread. No bound semantics
        // attached to it -- a stand-pat is an eval, not an alpha/beta
        // certificate. See docs/gpu-async-eval/consultative-eval-measurements.md.
        std::int16_t gpu_score;
        std::uint8_t gpu_depth, gpu_age;
        if (gpu_eval::active() &&
            gpu_eval::shared_gpu_tt().probe(board.get_hash(), gpu_score, gpu_depth, gpu_age) &&
            gpu_age == gpu_eval::shared_gpu_tt().current_age())
        {
            gpu_eval::shared_gpu_tt().record_useful_hit();
            stand_pat = gpu_score;

            // Measurement mode (CHESS26_GPU_MEASURE=1): of the GPU
            // scores the search actually reads, how many CHANGE what this
            // node does? A score that lands on the same side of beta as
            // the eval it replaced delivered nothing, however accurate it
            // was. This is the number that says whether the subsystem
            // should chase volume or selectivity -- usage_ratio_percent()
            // only says the score was read, not that it mattered.
            //
            // Off by default: it costs the very NNUE eval the GPU score
            // was there to avoid, so it is a diagnostic, not a feature.
            if (gpu_eval::measure_diagnostics())
            {
                const int nnue = Eval::eval_relative<Us>(board, alpha, beta);
                gpu_eval::shared_gpu_tt().record_decision_flip((gpu_score >= beta) != (nnue >= beta));
            }
        }
        else
        {
            stand_pat = Eval::eval_relative<Us>(board, alpha, beta);
        }
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

    int best_score = in_check ? -engine_constants::eval::Inf : stand_pat;
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

        if (!board.template is_move_legal<Us>(m))
            continue;

        board.play<Us>(m);

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
            int score = -engine_constants::eval::MateScore + ply;
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
        score = engine_constants::eval::MvvLvaTable[PAWN][PAWN];
    }
    else
    {
        score = engine_constants::eval::MvvLvaTable[move.get_to_piece()][move.get_from_piece()];
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

int SearchWorker::score_quiet_history(int raw_score, const Move &move, const Move &prev_move, const Move &prev_prev_move, Color us) const
{
    int score = raw_score;

    // Continuation history 1-ply : indexed by previous move's piece and destination
    if (prev_move != 0)
    {
        const int prev_piece = prev_move.get_from_piece();
        const int prev_to = prev_move.get_to_sq();
        const int move_to = move.get_to_sq();
        score += continuation_hist_1[us][prev_piece][prev_to][move_to] / 2;
    }

    // Continuation history 2-ply : indexed by 2-moves-ago previous move
    if (prev_prev_move != 0)
    {
        const int prev_prev_piece = prev_prev_move.get_from_piece();
        const int prev_prev_to = prev_prev_move.get_to_sq();
        const int move_to = move.get_to_sq();
        score += continuation_hist_2[us][prev_prev_piece][prev_prev_to][move_to] / 4;
    }

    return score;
}

template int SearchWorker::qsearch<WHITE>(int alpha, int beta, int ply);
template int SearchWorker::qsearch<BLACK>(int alpha, int beta, int ply);