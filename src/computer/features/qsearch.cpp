#include "computer.hpp"
int Computer::qsearch(int alpha, int beta)
{
    check_time();
    if (time_up)
        return eval::eval_relative(board.get_side_to_move(), board);

    q_nodes++;
    total_nodes++;

    // 1. Sondage de la Transposition Table (TT)
    int tt_score;
    Move tt_move;
    bool tt_hit = tt.probe(board.get_hash(), 0, 0, alpha, beta, tt_score, tt_move);
    if (tt_hit)
        return tt_score;

    // 2. Standing Pat (Évaluation statique)
    // On considère que ne rien faire est une option.
    int stand_pat = eval::eval_relative(board.get_side_to_move(), board);
    if (stand_pat >= beta)
        return beta;
    if (stand_pat > alpha)
        alpha = stand_pat;

    const Color player = board.get_side_to_move();

    // 3. Essai du TT Move (si c'est une capture/promotion)
    if (tt_move.get_value() != 0)
    {
        uint32_t flags = tt_move.get_flags();
        if (tt_move.get_to_piece() != NO_PIECE || flags == Move::EN_PASSANT_CAP || (flags & Move::PROMOTION_MASK))
        {
            board.play(tt_move);
            if (!MoveGen::is_king_attacked(board, player))
            {
                int score = -qsearch(-beta, -alpha);
                board.unplay(tt_move);

                if (score >= beta)
                    return beta;
                if (score > alpha)
                {
                    alpha = score;
                    stand_pat = score; // On met à jour le score de base
                }
            }
            else
                board.unplay(tt_move);
        }
    }

    // 4. Génération des captures
    MoveList list;
    MoveGen::generate_pseudo_legal_captures(board, player, list);

    // Scoring MVV-LVA pour le tri
    for (int i = 0; i < list.count; ++i)
        list.scores[i] = score_capture(list[i]);

    int best_score = stand_pat;
    int alpha_orig = alpha;
    Move best_move;

    // 5. Boucle des captures
    for (int i = 0; i < list.count; ++i)
    {
        Move &m = list.pick_best_move(i);
        if (m == tt_move)
            continue;

        check_time();
        if (time_up)
            break;

        Piece target = (m.get_flags() == Move::EN_PASSANT_CAP) ? PAWN : m.get_to_piece();
        int victim_val = eval::get_piece_score(target);
        int attacker_val = eval::get_piece_score(m.get_from_piece());

        // --- DELTA PRUNING ---
        // Si même avec un bonus (capture + promotion possible), on n'atteint pas alpha, on ignore.
        int promo_bonus = (m.get_flags() & Move::PROMOTION_MASK) ? 800 : 0;
        if (stand_pat + victim_val + promo_bonus + 200 < alpha)
            continue;

        // --- SEE PRUNING ---
        // On n'appelle le SEE que si on risque de perdre du matériel (Attaquant > Victime)
        // Sinon, le premier échange est forcément >= 0.
        if (attacker_val > victim_val || (m.get_flags() & Move::PROMOTION_MASK))
        {
            if (see(m.get_to_sq(), target, m.get_from_piece(), player, m.get_from_sq()) < 0)
                continue;
        }

        board.play(m);
        if (MoveGen::is_king_attacked(board, player))
        {
            board.unplay(m);
            continue;
        }

        int score = -qsearch(-beta, -alpha);
        board.unplay(m);

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

    // Sauvegarde TT et retour
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    tt.store(board.get_hash(), 0, 0, best_score, flag, best_move);

    return best_score;
}

int Computer::score_capture(const Move &move) const
{
    int attacker = move.get_from_piece();

    // 2. Identifier la victime (Qui est mangé ?)
    int victim_val = 0;

    if (move.get_flags() == Move::EN_PASSANT_CAP)
    {
        victim_val = eval::get_piece_score(PAWN);
    }
    else
    {
        // On regarde sur la case d'arrivée
        const int victim = move.get_to_piece();
        if (victim != NO_PIECE)
        {
            victim_val = eval::get_piece_score(victim);
        }
    }

    // 3. Calcul du score
    int attacker_val = eval::get_piece_score(attacker);

    // Formule MVV-LVA standard
    int score = 1000000 + (victim_val * 10 - attacker_val);

    // Bonus Promotion
    if (move.get_flags() == Move::PROMOTION_MASK)
    {
        score += eval::get_piece_score(QUEEN);
    }

    return score;
}
