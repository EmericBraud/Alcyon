#include "engine/search/worker.hpp"

#include "common/logger.hpp"

#include "engine/config/config.hpp"
#include "engine/config/eval.hpp"
#include "engine/utils/random.hpp"
#include "engine/engine_manager.hpp"
#include "worker.hpp"

template <Color Us>
int SearchWorker::score_move(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const
{
    const uint32_t move_val = move.get_value();
    if (move_val == tt_move.get_value())
        return 9600;

    const Piece from_piece = move.get_from_piece();
    const Piece to_piece = move.get_to_piece();
    const uint32_t flags = move.get_flags();

    // 1. Tactiques (Captures & Promotions)
    if (to_piece != NO_PIECE || flags == Move::Flags::EN_PASSANT_CAP || move.is_promotion())
    {
        const Piece target = (flags == Move::Flags::EN_PASSANT_CAP) ? PAWN : to_piece;
        const int mvv_lva = engine::config::eval::MvvLvaTable[target][from_piece];

        if (move.is_promotion())
        {
            // Priorise la promotion Dame
            const int promo_piece = move.get_promo_piece();
            if (promo_piece == QUEEN)
                return 9300;
            if (promo_piece == KNIGHT)
                return 8050;
            if (promo_piece == ROOK)
                return 8000;
            if (promo_piece == BISHOP)
                return 8000;
        }

        // On n'appelle SEE que si c'est potentiellement perdant (LVA prend MVV)
        if (Eval::get_piece_score(from_piece) > Eval::get_piece_score(target))
        {
            if (see<Us>(move.get_to_sq(), target, from_piece, move.get_from_sq()) < 0)
                return -1000 + mvv_lva; // Clairement perdant
        }

        return 8600 + mvv_lva; // Captures normales/gagnantes
    }

    // 2. Coups calmes prioritaires
    if (move_val == killer_moves[ply][0].get_value())
        return 8000;
    if (move_val == killer_moves[ply][1].get_value())
        return 8000;

    // Counter-move
    if (ply > 0 && prev_move != 0 && move_val == counter_moves[Us][prev_move.get_from_piece()][prev_move.get_to_sq()].get_value())
        return 7500;

    // Bonus spécial pour les échecs "calmes" (Crucial pour mat en 11)
    // Attention : nécessite que ta MoveGen ou une fonction légère détecte l'échec
    // if (gives_check(move)) return 6000000;

    // 3. History Moves (Score relatif)
    return history_moves[Us][move.get_from_sq()][move.get_to_sq()];
}
std::string SearchWorker::get_pv_line(int depth)
{
    std::string pv_line = "";
    std::vector<Move> moves_to_unplay;
    std::vector<uint64_t> visited_hashes;

    for (int i = 0; i < depth; i++)
    {
        Move m = shared_tt.get_move(board.get_hash());

        // 1. Si le coup est 0 (Nœud terminal, Mat, ou pas d'entrée), on arrête.
        if (m.get_value() == 0)
            break;

        // 2. Vérification de pseudo-légalité AVANT de toucher au board
        // Cela évite les asserts ou crashs dans is_move_legal si m est corrompu
        if (!board.is_move_pseudo_legal(m))
            break;

        if (!board.is_move_legal(m))
            break;

        // Détection de cycle
        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (uint64_t v : visited_hashes)
            if (v == h)
            {
                cycle_detected = true;
                break;
            }
        if (cycle_detected)
            break;

        visited_hashes.push_back(h);

        pv_line += m.to_uci() + " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

    for (int i = (int)moves_to_unplay.size() - 1; i >= 0; i--)
    {
        board.unplay(moves_to_unplay[i]);
    }
    return pv_line;
}

int SearchWorker::negamax_with_aspiration(int depth, int last_score)
{
    int delta = (depth >= 12) ? 100 : (depth >= 8) ? 50
                                                   : 16;
    int alpha = -engine::config::eval::Inf;
    int beta = engine::config::eval::Inf;

    if (depth >= 5)
    {
        alpha = last_score - delta;
        beta = last_score + delta;
    }

    int iterations = 0;
    const int max_iterations = 5;

    while (true)
    {
        ++iterations;
        int score = negamax(depth, alpha, beta, 0);

        if (abs(score) >= engine::config::eval::MateScore - depth)
        {
            return score;
        }
        if (abs(score) >= engine::config::eval::MateScore - 256)
        {
            alpha = -engine::config::eval::MateScore;
            beta = engine::config::eval::MateScore;
            continue;
        }

        if (shared_stop.load(std::memory_order_relaxed))
            return score;

        if (manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
            return score;
            // 2. Vérification de pseudo-légalité AVANT de toucher au board
            // Cela évite les asserts ou crashs dans is_move_legal si m est corrompu
        }

        // Succès : score dans la fenêtre
        if (score > alpha && score < beta)
        {
            best_root_move = out_move;
            return score;
        }

        // Ajustement delta intelligent
        if (score <= alpha)
        {
            // Fail-low
            delta = std::max(delta * 2, 50);
            alpha = std::max(-engine::config::eval::Inf, alpha - delta);
            if (thread_id == 0)
                logs::debug << "info string fail low" << std::endl;
        }
        else if (score >= beta)
        {
            // Fail-high
            delta = std::max(delta * 2, 50);
            beta = std::min(engine::config::eval::Inf, beta + delta);
            if (thread_id == 0)
                logs::debug << "info string fail high" << std::endl;
        }

        // Sécurité
        if (iterations >= max_iterations || delta > 2000)
        {
            alpha = -engine::config::eval::Inf;
            beta = engine::config::eval::Inf;
        }
    }
}

void SearchWorker::iterative_deepening()
{
    int last_score = 0;
    for (int depth = 1; depth < engine::config::search::MaxDepth; ++depth)
    {
        age_history();
        last_score = negamax_with_aspiration(depth, last_score);
        if (shared_stop.load(std::memory_order_relaxed))
            return;
        if (thread_id == 0)
        {
            auto elapsed_ms = std::max<long long>(1,
                                                  std::chrono::duration_cast<std::chrono::milliseconds>(
                                                      std::chrono::steady_clock::now() - start_time_ref)
                                                      .count());

            long long nodes = global_nodes.load(std::memory_order_relaxed);
            long long nps = nodes * 1000 / elapsed_ms;
            logs::uci
                << "info depth " << depth
                << " score cp " << last_score
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << shared_tt.get_hashfull()
                << " pv " << get_pv_line(depth)
                << std::endl;
        }
    }
    if (thread_id == 0)
        shared_stop.store(true, std::memory_order_relaxed);
}

bool SearchWorker::check_stop()
{
    if ((local_nodes & 32767) == 0)
    {
        global_nodes.fetch_add(local_nodes, std::memory_order_relaxed);
        local_nodes = 0;

        if (thread_id == 0 && manager.should_stop())
        {
            shared_stop.store(true, std::memory_order_relaxed);
        }
        if (shared_stop.load(std::memory_order_relaxed))
            return true;
    }
    ++local_nodes;
    return false;
}

template int SearchWorker::score_move<WHITE>(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;
template int SearchWorker::score_move<BLACK>(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;