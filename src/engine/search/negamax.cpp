#include "worker.hpp"
#include <atomic>

#include "engine/utils/random.hpp"
#include "engine/engine_manager.hpp"
#include "engine/search/move_picker.hpp"

#include "common/fatal.hpp"

#include <cstdlib>

namespace search
{
    // Evaluation statique du noeud, calculee AU PLUS UNE FOIS par noeud et
    // memorisee dans la pile de recherche (voir SearchWorker).
    //
    // La fenetre passee a prune_eval_relative est ignoree par les deux
    // builds (NNUE : eval() ne la lit pas ; HCE : lazy_eval_relative ne la
    // prend meme pas), donc la valeur ne depend que de la position et le
    // cache est exact. Si une eval paresseuse fenetree revenait un jour, ce
    // cache deviendrait faux -- d'ou la fenetre infinie explicite ici, qui
    // documente l'hypothese au lieu de la cacher.
    template <Color Us>
    inline int node_static_eval(SearchWorker &worker, int ply)
    {
        int &slot = worker.static_eval_stack[ply];
        if (slot == SearchWorker::kEvalNone)
            slot = Eval::prune_eval_relative<Us>(worker.get_board(),
                                                 -engine_constants::eval::Inf,
                                                 engine_constants::eval::Inf);
        return slot;
    }

    // "Notre position s'ameliore-t-elle ?" -- l'eval de ce noeud comparee a
    // celle de l'ancetre ply-2, qui est le dernier noeud ou c'etait deja
    // notre trait.
    //
    // Renvoie true quand l'information manque (racine, ancetre en echec, ou
    // ancetre qui n'a jamais eu besoin d'evaluer : le remplissage est
    // paresseux). C'est la convention de Stockfish, et elle est deliberee --
    // "improving" est la valeur qui elague le MOINS du cote alpha, donc
    // l'inconnu ne se paie pas en coups jetes.
    inline bool is_improving(const SearchWorker &worker, int ply)
    {
        if (ply < 2)
            return true;
        const int cur = worker.static_eval_stack[ply];
        const int prev = worker.static_eval_stack[ply - 2];
        if (cur == SearchWorker::kEvalNone || prev == SearchWorker::kEvalNone)
            return true;
        return cur > prev;
    }

    inline bool is_null(const VBoard &board, int ply)
    {
        if (ply == 0)
            return false;
        return board.is_repetition() || board.get_halfmove_clock() >= 100;
    }

    template <Color Us>
    inline bool razoring(SearchWorker &worker, int depth, int alpha, bool is_pv, bool in_check, int ply)
    {
        if (in_check || is_pv || depth > engine_constants::search::razoring::MaxDepth || ply == 0)
            return false;

        const int static_eval = node_static_eval<Us>(worker, ply);
        int margin = engine_constants::search::razoring::MarginDepthFactor * depth + engine_constants::search::razoring::MarginConst;
        return static_eval + margin <= alpha;
    }

    inline bool should_qsearch(int &depth, int ply, bool in_check)
    {
        if (depth > 0)
            return false;
        if (in_check && ply < engine_constants::search::MaxDepth - 5)
        {
            depth = 1;
            return false;
        }
        return true;
    }

    template <Color Us>
    inline bool reverse_futility_pruning(SearchWorker &worker, int depth, int ply, bool in_check, bool is_pv, int beta, int &return_score)
    {
        namespace rfp = engine_constants::search::reverse_futility_pruning;
        if (depth <= rfp::MaxDepth && !in_check && ply > 0 && !is_pv)
        {
            // Un seul test, sur le reseau complet (voir
            // Eval::prune_eval_relative). La seconde passe qui existait ici
            // n'avait d'objet que pour rattraper la tete PSQT ; elle
            // disparait avec elle.
            const int static_eval = node_static_eval<Us>(worker, ply);

            // improving RETIRE de la marge ici. Le RFP echoue haut : il
            // parie que la position est deja si bonne qu'aucun coup ne la
            // fera passer sous beta. Si en plus elle s'ameliore depuis deux
            // plys, le pari est plus sur, donc on exige moins de marge et on
            // coupe plus souvent. C'est le "depth - improving" de Stockfish.
            const int improving = is_improving(worker, ply) ? engine_constants::search::reverse_futility_pruning::ImprovingDepthBonus : 0;
            const int margin = rfp::MarginDepthFactor * (depth - improving) + rfp::MarginConst;
            if (static_eval - margin >= beta)
            {
                // Fail-HARD, comme avant : on rend beta, pas static_eval.
                // Le fail-soft est une amelioration separee (bornes TT plus
                // informatives) qui deplace l'arbre a elle seule -- mesure a
                // +20 % de noeuds a profondeur fixe -- donc elle merite son
                // propre SPRT et n'a rien a faire dans le lot improving.
                return_score = beta;
                return true;
            }
        }
        return false;
    }

    inline bool should_use_tt(bool tt_hit, int ply, bool is_pv, TTFlag flag, int tt_score, int beta)
    {
        if (tt_hit && ply > 0)
        {
            if (!is_pv)
            {
                return true;
            }
            else
            {

                if (flag == TT_EXACT ||
                    (flag == TT_BETA && tt_score >= beta))
                {
                    return true;
                }
            }
        }
        return false;
    }

    // Internal Iterative Reductions, en remplacement de l'IID.
    //
    // Les deux partent du meme constat -- pas de coup TT ici -- et en tirent
    // la conclusion inverse. L'IID DEPENSAIT une recherche a depth - 4 pour
    // se fabriquer un coup d'ordonnancement. L'IIR ECONOMISE un ply : un
    // noeud sans coup TT n'a jamais ete recherche a profondeur suffisante
    // pour en laisser un, il est donc statistiquement moins important, et le
    // temps rendu profite au reste de l'arbre.
    //
    // Restreint aux noeuds PV et cut (`!all_node`), comme Stockfish. Il lui
    // reste une condition que nous n'avons pas, `!followPV`, qui protege la
    // ligne principale de l'iteration precedente d'une perte cumulee : elle
    // demande de memoriser cette PV indexee par ply, ce qu'on ne fait pas
    // encore. Notre IIR reste donc un peu plus agressif que la reference, dont
    // le commentaire porte la note (*Scaler) "Making IIR more aggressive
    // scales poorly".
    inline void internal_iterative_reduction(Move tt_move, bool all_node, int &depth)
    {
        if (!all_node && tt_move == 0 && depth >= engine_constants::search::internal_iterative_reduction::MinDepth)
            depth -= engine_constants::search::internal_iterative_reduction::Reduction;
    }

    template <Color Us>
    inline bool nmp(SearchWorker &worker, int depth, int ply, bool allow_null, bool in_check, bool is_mate_node, int alpha, int beta, int &return_score)
    {
        // Garde zugzwang : sans piece autre que les pions, "passer son tour"
        // n'est pas une borne inferieure du vrai coup -- en finale de pions
        // le trait est souvent un desavantage, et le null move prouve alors
        // des coupures fausses. Correctness, pas vitesse.
        const U64 non_pawn = worker.get_board().get_occupancy<Us>() &
                             ~worker.get_board().template get_piece_bitboard<Us, PAWN>() &
                             ~worker.get_board().template get_piece_bitboard<Us, KING>();
        if (depth >= engine_constants::search::null_move_pruning::MinDepth && ply > 0 && allow_null && !in_check && !is_mate_node && beta < 9000 && alpha > -9000 && non_pawn != 0)
        {
            int stored_ep;
            worker.get_tt().prefetch(worker.get_board().get_hash());
            worker.get_board().play_null_move(stored_ep);
            int R = engine_constants::search::null_move_pruning::RConst + depth / engine_constants::search::null_move_pruning::RDiv;
            R = std::min(R, depth - 1);
            int score = -worker.negamax<!Us>(depth - 1 - R, -beta, -beta + 1, ply + 1, false, false);
            worker.get_board().unplay_null_move(stored_ep);

            if (score >= beta)
            {
                return_score = (score >= engine_constants::eval::MateScore - engine_constants::search::MaxDepth) ? beta : score;
                return true;
            }
        }
        return false;
    }

    template <Color Us>
    bool should_futility_pruning(SearchWorker &worker, int depth, int ply, bool in_check, bool is_pv, bool is_mate_node, int alpha)
    {
        if (depth <= engine_constants::search::futility_pruning::MaxDepth && !in_check && !is_pv && ply > 0 && !is_mate_node)
        {
            // improving AJOUTE a la marge ici -- signe oppose au RFP, et ce
            // n'est pas une incoherence. La futility echoue BAS : elle jette
            // des coups calmes en pariant qu'aucun ne remontera jusqu'a
            // alpha. Une position qui s'ameliore rend ce pari moins sur,
            // donc on exige une marge plus grande et on jette moins.
            const int improving = is_improving(worker, ply) ? engine_constants::search::futility_pruning::ImprovingDepthBonus : 0;
            const int futil_margin = engine_constants::search::futility_pruning::MarginConst + engine_constants::search::futility_pruning::MarginDepthFactor * (depth + improving);
            const int static_eval = node_static_eval<Us>(worker, ply);

            if (static_eval + futil_margin <= alpha)
            {
                return true;
            }
        }
        return false;
    }

    template <Color Us>
    inline bool is_singular_search(SearchWorker &worker, Move tt_move, int depth, int ply, bool in_check, bool cut_node, Move excluded_move, Move m)
    {
        if (!in_check && depth >= engine_constants::search::singular::MinDepth && ply > 0 && m == tt_move && excluded_move == 0)
        {
            TTFlag ttf;
            int tts;
            Move ttm;
            if (worker.get_tt().probe(worker.board.get_hash(), depth, ply, -engine_constants::eval::Inf, engine_constants::eval::Inf, tts, ttm, ttf))
            {
                if (ttf == TT_EXACT || ttf == TT_ALPHA)
                {
                    int singular_beta = tts - (depth * 2);
                    int singular_depth = (depth - 1) / 2;
                    int score = worker.negamax<Us>(singular_depth, singular_beta - 1, singular_beta, ply, false, cut_node, m);

                    if (score < singular_beta)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    bool should_lmp(bool in_check, int depth, bool is_tactical, int moves_searched, bool improving)
    {
        if (!in_check && depth <= engine_constants::search::null_move_reduction::MaxDepth && !is_tactical)
        {
            int max_moves = engine_constants::search::null_move_reduction::MaxMovesConst + (depth * depth * engine_constants::search::null_move_reduction::MaxMovesDepthSqFactor);
            // Position qui ne s'ameliore pas : on s'autorise a regarder
            // moins de coups avant de couper la liste. Stockfish divise par
            // deux ; le diviseur est tunable, et 1 rend le comportement
            // d'avant improving.
            if (!improving)
                max_moves /= engine_constants::search::null_move_reduction::NotImprovingDiv;
            if (moves_searched >= max_moves)
            {
                return true;
            }
        }
        return false;
    }

    template <Color Us>
    inline bool should_see_pruning(SearchWorker &worker, bool in_check, bool is_pv, int depth, int moves_searched, Move tt_move, Move m)
    {

        if (!in_check && !is_pv &&
            depth <= engine_constants::search::see_pruning::MaxDepth &&
            moves_searched > 1 &&
            m != tt_move &&
            m.is_capture())
        {
            if (Eval::get_piece_score(m.get_from_piece()) > Eval::get_piece_score(m.get_to_piece()))
            {
                int see_score = worker.see<Us>(
                    m.get_to_sq(),
                    m.get_to_piece(),
                    m.get_from_piece(),
                    m.get_from_sq());

                int threshold = -engine_constants::search::see_pruning::ThresholdDepthFactor * depth - Eval::get_piece_score(m.get_to_piece()) / 2;

                if (see_score < threshold)
                    return true;
            }
        }
        return false;
    }

    template <Color Us>
    inline bool late_move_reduction_search(SearchWorker &worker, int depth, int ply, bool in_check, bool is_tactical, int moves_searched, int extension, bool cut_node, bool improving, Move tt_move, Move m, Move prev_m, Move prev_prev_m, int alpha, int &score)
    {
        if (depth >= engine_constants::search::late_move_reduction::MinDepth && moves_searched >= engine_constants::search::late_move_reduction::MinMovesSearched && !is_tactical && !in_check && extension == 0)
        {
            int r = static_cast<int>(worker.lmr_table[std::min(depth, 63)][std::min(moves_searched, 63)]);
            if (tt_move == 0)
                r += engine_constants::search::late_move_reduction::NoTTMoveBonus;
            r += engine_constants::search::late_move_reduction::CutNodeBonus * cut_node;
            // Position qui ne s'ameliore pas : rien n'indique que ces coups
            // tardifs meritent leur profondeur, on reduit plus fort.
            r += engine_constants::search::late_move_reduction::NotImprovingBonus * !improving;

            // Modulation par l'history du COUP. Jusqu'ici r ne regardait que
            // la position du coup dans la liste : deux coups calmes au meme
            // rang etaient reduits pareil, que l'un ait coupe cent fois dans
            // cette partie et l'autre jamais. Les tables savaient les
            // distinguer -- elles servaient a l'ORDONNANCEMENT -- mais la
            // recherche jetait l'information juste apres s'en etre servie.
            //
            // On recalcule ici plutot que de relire list.scores[] : ce
            // tableau porte le bruit de diversification SMP (+/-1024 pour
            // les threads != 0, voir MovePicker) et des constantes plates
            // pour les killers et counters (8000/7900/7500), qui ne sont pas
            // des valeurs d'history. Trois lectures de table, le prix est
            // nul devant un noeud.
            const int hist = worker.score_quiet_history(
                worker.history_moves[Us][m.get_from_sq()][m.get_to_sq()],
                m, prev_m, prev_prev_m, Us);
            r -= hist / engine_constants::search::late_move_reduction::HistDivisor;

            r = std::clamp(r, 0, depth - engine_constants::search::late_move_reduction::MaxDepthReduction);

            // Sonde speculative : on ATTEND son echec, donc on declare
            // l'enfant noeud cut quel que soit le parent, pour qu'il elague
            // plus dur. La re-recherche ci-dessous rattrape si on se trompe.
            score = -worker.negamax<!Us>(depth - 1 - r, -alpha - 1, -alpha, ply + 1, true, true);

            // Re-search si le coup réduit semble bon
            if (score > alpha)
                score = -worker.negamax<!Us>(depth - 1, -alpha - 1, -alpha, ply + 1, true, !cut_node);
            return true;
        }
        return false;
    }
}

inline TableBase::WDL_Result should_tb_probe(const Board &board, TableBase &shared_tb)
{
    return shared_tb.probe_wdl(board);
}

inline int wdl_score(TableBase::WDL_Result r, int ply)
{
    switch (r)
    {
    case TableBase::WDL_Result::LOSS:
        return -(engine_constants::eval::SyzygyScore - ply);
    case TableBase::WDL_Result::CURSED_WIN:
    case TableBase::WDL_Result::DRAW:
    case TableBase::WDL_Result::BLESSED_LOSS:
        return 0;
    case TableBase::WDL_Result::WIN:
        return engine_constants::eval::SyzygyScore - ply;
    default:
        FATAL("Incoherent WDL value returned");
    }
}

template <Color Us>
int SearchWorker::negamax(int depth, int alpha, int beta, int ply, bool allow_null, bool cut_node, Move excluded_move)
{

    // =============================== Quick return cases ===============================
    if (check_stop())
        return alpha;

    if (max_extended_depth < ply)
        max_extended_depth = ply;

    if (search::is_null(board, ply))
        return (board.get_history_size() < 20) ? -25 : 0;

    if (ply > 0 &&
        board.get_halfmove_clock() == 0 &&
        board.get_castling_rights() == 0 &&
        std::popcount(board.get_occupancy<NO_COLOR>()) <= engine_constants::eval::SyzygyMaxPieces)
    {
        TableBase::WDL_Result r_tb = should_tb_probe(board, shared_tb);
        if (r_tb != TableBase::WDL_Result::FAIL)
            return wdl_score(r_tb, ply);
    }

    if (ply >= engine_constants::search::MaxDepth)
        return Eval::lazy_eval_relative<Us>(board);

    // Ce ply est reutilise par toutes les branches deja explorees a cette
    // profondeur : l'eval memorisee appartient a une AUTRE position. On
    // invalide avant le premier mecanisme qui pourrait la lire.
    static_eval_stack[ply] = kEvalNone;

    const bool is_pv = (beta - alpha > 1);
    // Classification de Knuth-Moore. Un noeud all doit epuiser tous ses coups
    // pour PROUVER qu'aucun ne passe : lui retirer de la profondeur affaiblit
    // une preuve qu'on devra croire. Et les noeuds all sont le gros de l'arbre,
    // donc c'est la que la composition de l'IIR le long des chaines de noeuds
    // sans coup TT frappe le plus fort. Stockfish les exclut, nous aussi.
    const bool all_node = !(is_pv || cut_node);
    const bool in_check = board.is_king_attacked<Us>();
    const bool is_mate_node = (alpha < engine_constants::eval::MateScore && beta > -engine_constants::eval::MateScore && in_check);

    if (search::razoring<Us>(*this, depth, alpha, is_pv, in_check, ply))
        return qsearch<Us>(alpha, beta, ply);

    Move tt_move = 0;
    {
        TTFlag flag;
        int tt_score;
        bool tt_hit = shared_tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move, flag);
        if (search::tt_cutoffs_enabled() && tt_move != excluded_move &&
            search::should_use_tt(tt_hit, ply, is_pv, flag, tt_score, beta))
            return tt_score;
    }

    if (search::should_qsearch(depth, ply, in_check))
        return qsearch<Us>(alpha, beta, ply);

    {
        // Fail-hard conserve : le helper rend beta (voir son commentaire).
        // Le score passe par une sortie pour laisser la place au fail-soft,
        // qui fera l'objet d'un SPRT separe.
        int rfp_score;
        if (search::reverse_futility_pruning<Us>(*this, depth, ply, in_check, is_pv, beta, rfp_score))
            return rfp_score;
    }

    // =============================== Search ===============================

    {
        int return_score;
        if (search::nmp<Us>(*this, depth, ply, allow_null, in_check, is_mate_node, alpha, beta, return_score))
            return return_score;
    }

    // Place APRES le null move, comme Stockfish (son etape 11 suit l'etape
    // 10) : le NMP calcule sa reduction sur la profondeur pleine, l'IIR
    // reduit ensuite ce qui reste a explorer.
    search::internal_iterative_reduction(tt_move, all_node, depth);

    const bool futil_pruning = search::should_futility_pruning<Us>(*this, depth, ply, in_check, is_pv, is_mate_node, alpha);

    // Calcule APRES razoring/RFP/futility, qui sont les mecanismes capables
    // de remplir static_eval_stack[ply] : le lire avant reviendrait a lire
    // kEvalNone et a retomber systematiquement sur le defaut "true".
    const bool improving = search::is_improving(*this, ply);

    const auto *history = board.get_history();
    const Move prev_m = ply > 0 ? history->back().move : 0;
    const Move prev_prev_m = (history->size() >= 2) ? (*history)[history->size() - 2].move : 0;
    MovePicker list(board, tt_move, ply, prev_m, thread_id);

    // 7. PVS Loop (Principal Variation Search)
    int alpha_orig = alpha;

    int best_score = -engine_constants::eval::Inf;
    Move best_move_this_node = 0;
    int moves_searched = 0;

    while (true)
    {
        Move m = list.pick_next<Us>(*this);

        if (m == 0) // No moves left
            break;

        if (!board.is_move_legal<Us>(m))
            continue;

        if (m == excluded_move)
            continue;

        shared_tt.prefetch(board.get_hash_after(m));
        bool is_singular = search::is_singular_search<Us>(*this, tt_move, depth, ply, in_check, cut_node, excluded_move, m);

        int score;
        const bool is_tactical = list.current_is_tactical;
        if (search::should_lmp(in_check, depth, is_tactical, moves_searched, improving))
            continue;

        if (futil_pruning && moves_searched >= 1 && !is_tactical)
        {
            if (!board.template gives_check<Us>(m))
                continue;
        }
        if (search::should_see_pruning<Us>(*this, in_check, is_pv, depth, moves_searched, tt_move, m))
            continue;

        ++moves_searched;

        board.play<Us>(m);

        bool gives_check = board.is_king_attacked<!Us>();

        int extension = 0;
        if (gives_check && depth >= 2)
            extension = 1;
        if (is_singular)
            extension = 1;
        int new_depth = depth - 1 + extension;

        if (ply + new_depth >= engine_constants::search::MaxDepth)
            new_depth = engine_constants::search::MaxDepth - ply;

        if (!search::late_move_reduction_search<Us>(*this, depth, ply, in_check, is_tactical, moves_searched, extension, cut_node, improving, tt_move, m, prev_m, prev_prev_m, alpha, score))
        {
            if (moves_searched > 1) // Null Window Search pour PVS
            {
                score = -negamax<!Us>(new_depth, -alpha - 1, -alpha, ply + 1, true, !cut_node);
            }
            else // Full Window Search (seulement si moves_searched == 1 (donc pas de TT move))
            {
                score = -negamax<!Us>(new_depth, -beta, -alpha, ply + 1, true, false);
            }
        }

        // Si le score est dans la fenêtre mais pas une coupure, on re-cherche normalement
        if (score > alpha && score < beta && moves_searched > 1)
        {
            score = -negamax<!Us>(new_depth, -beta, -alpha, ply + 1, true, false);
        }

        board.unplay<Us>(m);

        // --- MISE À JOUR DES SCORES ET DES TABLES ---
        if (score >= beta)
        {
            if (search::order_stats_enabled())
                search::record_cutoff(moves_searched, is_tactical);
            shared_tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);

            if (!is_tactical)
            {

                // MALUS : On punit tous les coups calmes testés AVANT et qui ont échoué
                if (list.stage == PickerStages::QUIETS)
                {
                    int bonus = depth * depth;

                    // On récompense le coup gagnant
                    update_hist(history_moves[Us][m.get_from_sq()][m.get_to_sq()], bonus);

                    // Update continuation history
                    if (prev_m != 0)
                    {
                        const int prev_piece = prev_m.get_from_piece();
                        const int prev_to = prev_m.get_to_sq();
                        const int m_to = m.get_to_sq();
                        update_hist(continuation_hist_1[Us][prev_piece][prev_to][m_to], bonus);
                    }
                    if (prev_prev_m != 0)
                    {
                        const int prev_prev_piece = prev_prev_m.get_from_piece();
                        const int prev_prev_to = prev_prev_m.get_to_sq();
                        const int m_to = m.get_to_sq();
                        update_hist(continuation_hist_2[Us][prev_prev_piece][prev_prev_to][m_to], bonus);
                    }

                    for (int j = 0; j < list.index - 1; ++j)
                    {
                        Move failed_move = list.list.moves[j];
                        // On ne punit que les coups calmes (pas les captures/promotions)

                        update_hist(history_moves[Us][failed_move.get_from_sq()][failed_move.get_to_sq()], -bonus);
                    }
                }

                // 4. Update Killer Moves
                if (m.get_value() != killer_moves[ply][0].get_value())
                {
                    killer_moves[ply][1] = killer_moves[ply][0];
                    killer_moves[ply][0] = m;
                }
            }
            return score;
        }

        if (score > best_score)
        {
            best_score = score;
            best_move_this_node = m;
            if (score > alpha)
            {
                alpha = score;
            }
        }
    }
    if (ply == 0)
    {
        this->out_move = best_move_this_node;
    }

    // 8. Gestion des Mats et Pats
    if (moves_searched == 0)
    {
        int score = in_check ? -engine_constants::eval::MateScore + ply : 0;
        shared_tt.store(board.get_hash(), depth, ply, score, TT_EXACT, 0);
        return score;
    }

    if (shared_stop.load(std::memory_order_relaxed)) // We don't write in TT if shared_stop
        return best_score;

    // 9. Sauvegarde TT Finale
    TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
    shared_tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

    return best_score;
}

template int SearchWorker::negamax<WHITE>(int depth, int alpha, int beta, int ply, bool allow_null, bool cut_node, Move excluded_move);
template int SearchWorker::negamax<BLACK>(int depth, int alpha, int beta, int ply, bool allow_null, bool cut_node, Move excluded_move);