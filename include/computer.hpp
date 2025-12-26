#pragma once
#include "pos_eval.hpp"

#define MAX_DEPTH 25
constexpr int INF = 1000000;

struct MoveScorer
{
    Move m;
    int score;
};

using Clock = std::chrono::steady_clock;

class Computer
{
    Board &board;
    TranspositionTable tt;
    std::array<Move, MAX_DEPTH> move_stack;
    // [Color] [From sq] [To sq]
    int history_moves[2][64][64];

    long long total_nodes = 0;
    long long tt_cuts = 0;
    long long beta_cutoffs = 0;
    long long q_nodes = 0;

    Move killer_moves[MAX_DEPTH][2];
    Clock::time_point start_time;
    int time_limit_ms = 0;
    bool time_up = false;

    int qsearch(int alpha, int beta)
    {
        check_time();
        if (time_up)
            return eval::eval_relative(board.get_side_to_move(), board);
        q_nodes++;
        total_nodes++; // On compte aussi ces noeuds

        // 1. Standing Pat (On suppose qu'on ne fait rien)
        // Si on ne capture pas, quel est le score ?
        // Cela sert de "borne inférieure" (Lower Bound).

        int tt_score;
        Move tt_move;
        bool tt_hit = tt.probe(board.get_hash(), 0, 0, alpha, beta, tt_score, tt_move);
        if (tt_hit)
        {
            return tt_score;
        }

        int alpha_orig = alpha;

        // Coupure Beta "Lazy" : Si la position actuelle est déjà trop forte, on coupe.
        int stand_pat = eval::eval_relative(board.get_side_to_move(), board);
        if (stand_pat >= beta)
            return beta;
        if (stand_pat > alpha)
            alpha = stand_pat;

        const int BIG_DELTA = 975;
        if (stand_pat < alpha - BIG_DELTA)
            return alpha;

        const Color player = board.get_side_to_move();
        MoveList list;
        MoveGen::generate_pseudo_legal_captures(board, player, list);

        for (int i = 0; i < list.count; ++i)
        {
            int score = score_capture(list[i]);
            // Bonus si c'est le coup suggéré par la TT
            if (tt_move.get_value() != 0 && list[i].get_value() == tt_move.get_value() && (list[i].get_flags() == Move::Flags::CAPTURE || list[i].get_flags() == Move::Flags::EN_PASSANT_CAP || list[i].get_flags() == Move::Flags::PROMOTION_MASK))
            {
                score += 2000000;
            }
            list.scores[i] = score;
        }

        int best_score = stand_pat;
        Move best_move;
        for (int i = 0; i < list.count; ++i)
        {
            Move &m = list.pick_best_move(i);
            check_time();
            if (time_up)
                break;
            int victim_val = 0;
            if (m.get_flags() == Move::EN_PASSANT_CAP)
            {
                victim_val = eval::get_piece_score(PAWN);
            }
            else
            {
                int victim = m.get_to_piece();
                if (victim != NO_PIECE)
                    victim_val = eval::get_piece_score(victim);
            }
            int attacker_val = eval::get_piece_score(m.get_from_piece());
            if (attacker_val == 0)
                attacker_val = eval::get_piece_score(board.get_piece_on_square(m.get_from_sq()).second);

            if (attacker_val > victim_val)
            {
                if (stand_pat + victim_val + 200 < alpha)
                    continue;
            }

            // Delta Pruning (Optimisation optionnelle mais recommandée)
            // Si ma capture ne suffit pas à remonter le score (ex: P x D alors que je perds une Dame), on ignore.
            // (Laissons ça de côté pour l'instant pour la stabilité)

            board.play(m);

            if (MoveGen::is_king_attacked(board, player))
            {
                board.unplay(m);
                continue;
            }

            // Appel récursif à qsearch (pas negamax !)
            int score = -qsearch(-beta, -alpha);

            board.unplay(m);

            if (score >= beta)
            {
                tt.store(board.get_hash(), 0, 0, beta, TT_BETA, m);
                return beta;
            }

            if (score > alpha)
            {
                alpha = score;
                best_score = score;
                best_move = m;
            }
        }

        TTFlag flag = (best_score <= alpha_orig) ? TT_ALPHA : TT_EXACT;
        tt.store(board.get_hash(), 0, 0, best_score, flag, best_move);

        return alpha;
    }
    inline void check_time()
    {
        if (time_up)
            return;

        auto now = Clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

        if (elapsed >= time_limit_ms)
        {
            time_up = true;
        }
    }

public:
    int negamax(int depth, int alpha, int beta, int ply)
    {
        if (ply > 0 && board.is_repetition())
        {
            return 0;
        }
        int tt_score;
        Move tt_move;
        total_nodes++;
        bool tt_hit = tt.probe(board.get_hash(), depth, ply, alpha, beta, tt_score, tt_move);

        if (tt_hit && ply > 0)
        {
            tt_cuts++;
            return tt_score;
        }
        check_time();
        if (time_up)
            return eval::eval_relative(board.get_side_to_move(), board);
        if (depth <= 0)
        {
            return qsearch(alpha, beta);
        }
        if (depth >= 3 && ply > 0 && !MoveGen::is_king_attacked(board, board.get_side_to_move()))
        {
            int stored_ep;
            board.play_null_move(stored_ep);
            int R = 2;
            int score = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1);

            board.unplay_null_move(stored_ep);

            if (score >= beta)
            {
                if (score >= MATE_SCORE - MAX_DEPTH)
                    return beta;
                return score;
            }
        }

        const Color player = board.get_side_to_move();
        MoveList list;
        MoveGen::generate_legal_moves(board, list);

        for (int i = 0; i < list.count; ++i)
        {
            list.scores[i] = score_move(list.moves[i], board, tt_move, ply);
        }

        int legal_moves_count = 0;
        int best_score = -INF;
        Move best_move_this_node;
        int alpha_orig = alpha;

        int moves_searched = 0;
        for (int i = 0; i < list.count; ++i)
        {
            check_time();
            if (time_up)
                break;
            Move &m = list.pick_best_move(i);
            const int m_score = list.scores[i];
            board.play(m);

            legal_moves_count++;
            moves_searched++;

            int score;
            bool needs_full_search = true;
            bool is_tactical = (m_score >= 900000);
            // LMR (Late Move Reduction)
            // Conditions :
            // 1. Profondeur suffisante (> 2)
            // 2. On a déjà testé les bons coups (moves_searched > 4)
            // 3. Ce n'est pas un coup tactique (pas capture, pas promotion)
            // 4. On n'est pas en échec (trop dangereux de réduire)
            // --- LMR (LATE MOVE REDUCTION) ---
            if (depth >= 3 && moves_searched > 4 &&
                ply > 0 &&
                !is_tactical &&
                !MoveGen::is_king_attacked(board, player))
            {
                int reduction = 1;
                if (depth > 6)
                    reduction = 2;
                score = -negamax(depth - 1 - reduction, -beta, -alpha, ply + 1);
                if (score <= alpha)
                {
                    // Move is bad even tho we searched a bit
                    // It is really a bad move
                    needs_full_search = false;
                }
            }
            // --- RECHERCHE COMPLÈTE (FULL DEPTH) ---
            // On la fait si :
            // A) On n'a pas fait de LMR (premiers coups, captures, échecs...)
            // B) Ou si le LMR a échoué (le coup semblait trop beau pour être vrai)
            if (needs_full_search)
            {

                score = -negamax(depth - 1, -beta, -alpha, ply + 1);
            }

            board.unplay(m);

            if (score >= beta)
            {
                beta_cutoffs++;
                tt.store(board.get_hash(), depth, ply, score, TT_BETA, m);
                if (m.get_to_piece() == NO_PIECE && m.get_flags() != Move::EN_PASSANT_CAP)
                {
                    int bonus = depth * depth;
                    history_moves[board.get_side_to_move()][m.get_from_sq()][m.get_to_sq()] += bonus;
                    if (!(m.get_value() == killer_moves[ply][0].get_value()))
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

        if (legal_moves_count == 0)
        {
            if (MoveGen::is_king_attacked(board, player)) // Checkmate
            {
                return -MATE_SCORE + ply;
            }
            else
            {
                return 0;
            }
        }
        TTFlag flag;
        if (best_score <= alpha_orig)
            flag = TT_ALPHA;
        else
            flag = TT_EXACT;

        tt.store(board.get_hash(), depth, ply, best_score, flag, best_move_this_node);

        return best_score;
    }

    void play(int time_ms = 5000)
    {
        time_limit_ms = time_ms;
        time_up = false;
        start_time = Clock::now();

        Move best_move;
        total_nodes = 0;
        tt_cuts = 0;
        beta_cutoffs = 0;
        clear_killers();
        age_history();

        for (int current_depth = 1; current_depth <= MAX_DEPTH; ++current_depth)
        {
            if (time_up)
                break;

            int alpha = -INF;
            int beta = INF;

            int score = negamax(current_depth, alpha, beta, 0);

            if (!time_up)
            {
                // On garde le dernier résultat COMPLET
                best_move = tt.get_move(board.get_hash());

                std::cout << "Depth " << current_depth
                          << " | Score " << score
                          << " | Nodes " << total_nodes
                          << " | PV: " << get_pv_line(current_depth)
                          << std::endl;
            }
        }

        board.play(best_move);
    }

    int score_move(const Move &move, const Board &board, const Move &tt_move, int ply)
    {
        if (tt_move.get_value() != 0 && move.get_value() == tt_move.get_value())
        {
            return 2000000;
        }

        const Piece from_piece{move.get_from_piece()};

        const Piece to_piece{move.get_to_piece()};
        const uint32_t flags{move.get_flags()};

        if (flags == Move::Flags::PROMOTION_MASK)
        {
            return 1000000 + (eval::get_piece_score(QUEEN) * 10);
        }

        if (to_piece != NO_PIECE)
        {
            int victim_val = eval::get_piece_score(to_piece);
            int attacker_val = eval::get_piece_score(from_piece);
            return 1000000 + (victim_val * 10 - attacker_val);
        }
        if (flags == Move::Flags::EN_PASSANT_CAP)
        {
            int victim_val = eval::get_piece_score(PAWN);
            int attacker_val = eval::get_piece_score(from_piece);
            return 1000000 + (victim_val * 10 - attacker_val);
        }
        if (move.get_value() == killer_moves[ply][0].get_value())
            return 900000;
        if (move.get_value() == killer_moves[ply][1].get_value())
            return 800000;

        if (to_piece == NO_PIECE)
        {
            return history_moves[board.get_side_to_move()][move.get_from_sq()][move.get_to_sq()];
        }

        return 0;
    }
    // Score uniquement pour le Quiescence Search (MVV-LVA)
    int score_capture(const Move &move) const
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
    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                    history_moves[c][f][t] /= 8;
    }

    std::string get_pv_line(int depth)
    {
        std::string pv_line = "";
        Board temp_board = board; // Copie locale pour ne pas corrompre le plateau réel

        for (int i = 0; i < depth; i++)
        {
            Move m = tt.get_move(temp_board.get_hash());
            if (m.get_value() == 0)
                break; // Plus de coup en cache

            pv_line += m.to_uci() + " ";
            temp_board.play(m);
        }
        return pv_line;
    }

public:
    Computer(Board &board) : board(board)
    {
        init_zobrist();
        tt.resize(1024);
        clear_killers();
        clear_history();
    }

    int eval_position()
    {
        return negamax(MAX_DEPTH, -INF, INF, 0);
    }
    void clear_killers()
    {
        std::memset(killer_moves, 0, sizeof(killer_moves));
    }
    void clear_history()
    {
        std::memset(history_moves, 0, sizeof(history_moves));
    }
};