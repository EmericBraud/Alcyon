#include "engine/search/worker.hpp"

#include <cstdio>
#include "common/logger.hpp"

#include "engine/config/config.hpp"
#include "engine/config/eval.hpp"
#include "engine/engine_manager.hpp"
#include "engine/eval/gpu/gpu_queue.hpp"
#include "worker.hpp"

#include <algorithm>
#include <limits>
#include <cstdlib>

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
        const int mvv_lva = engine_constants::eval::MvvLvaTable[target][from_piece];

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

    for (int i = 0; i < std::min(depth, 10); i++)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            break;

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

std::string SearchWorker::get_pv_line_with_root(Move root_move, int depth)
{
    if (root_move.get_value() == 0)
        return "";

    if (!board.is_move_pseudo_legal(root_move) || !board.is_move_legal(root_move))
        return "";

    std::string pv_line = root_move.to_uci();
    if (depth <= 1)
        return pv_line;

    pv_line += " ";
    board.play(root_move);
    auto guard = CHESS26_SCOPE_EXIT([&]()
                                    { board.unplay(root_move); });

    std::vector<Move> moves_to_unplay;
    std::vector<uint64_t> visited_hashes;

    for (int i = 0; i < std::min(depth - 1, 10); ++i)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            break;

        Move m = shared_tt.get_move(board.get_hash());
        if (m.get_value() == 0)
            break;
        if (!board.is_move_pseudo_legal(m) || !board.is_move_legal(m))
            break;

        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (uint64_t v : visited_hashes)
        {
            if (v == h)
            {
                cycle_detected = true;
                break;
            }
        }
        if (cycle_detected)
            break;

        visited_hashes.push_back(h);
        pv_line += m.to_uci();
        pv_line += " ";
        board.play(m);
        moves_to_unplay.push_back(m);
    }

    for (int i = static_cast<int>(moves_to_unplay.size()) - 1; i >= 0; --i)
        board.unplay(moves_to_unplay[i]);

    return pv_line;
}

void SearchWorker::maybe_submit_pv_leaf_to_gpu(Move pv_root, int depth)
{
    if (!gpu_eval::enabled.load(std::memory_order_relaxed))
        return;

    if (pv_root.get_value() == 0)
        return;
    if (!board.is_move_pseudo_legal(pv_root) || !board.is_move_legal(pv_root))
        return;

    // Walk down to the PV leaf on this worker's OWN board -- same
    // technique as get_pv_line_with_root, mirrored here instead of
    // reused because that one builds a UCI string (allocates), which
    // this doesn't need. `board` is back at the root here (negamax_
    // with_aspiration has just returned), same precondition as the
    // reporting call sites in iterative_deepening().
    //
    // IMPORTANT: use Board::play/unplay explicitly (the core-board layer),
    // NOT VBoard::play/unplay -- the latter also updates the NNUE
    // accumulator (or HCE eval_state), which this walk has no use for
    // (nothing here calls evaluate()) and would be a pure wasted cost on
    // the search hot path. Bitboards/zobrist/side-to-move are all this
    // needs, same as gpu_queue.cpp's separate `Board scratch`.
    // Fixed capacity: pv_root plus up to 10 more plies (the loop below
    // is bounded by std::min(depth - 1, 10)), so 11 is an exact bound --
    // no runtime allocation on this hot path.
    constexpr int kMaxPvWalkPlies = 11;
    std::array<Move, kMaxPvWalkPlies> moves_to_unplay;
    std::array<uint64_t, kMaxPvWalkPlies - 1> visited_hashes;
    int num_moves_to_unplay = 0;
    int num_visited_hashes = 0;

    board.Board::play(pv_root);
    moves_to_unplay[num_moves_to_unplay++] = pv_root;

    for (int i = 0; i < std::min(depth - 1, 10); ++i)
    {
        if (board.is_repetition() || board.get_halfmove_clock() >= 100)
            break;

        Move m = shared_tt.get_move(board.get_hash());
        if (m.get_value() == 0)
            break;
        if (!board.is_move_pseudo_legal(m) || !board.is_move_legal(m))
            break;

        uint64_t h = board.get_hash();
        bool cycle_detected = false;
        for (int v = 0; v < num_visited_hashes; ++v)
            if (visited_hashes[v] == h)
            {
                cycle_detected = true;
                break;
            }
        if (cycle_detected)
            break;

        visited_hashes[num_visited_hashes++] = h;
        board.Board::play(m);
        moves_to_unplay[num_moves_to_unplay++] = m;
    }

    // `board` is now at the PV leaf. Its own TT move (if any) is the best
    // available guess at which child the search will look at first -- see
    // submit_current_position_to_gpu's note on why that matters.
    submit_current_position_to_gpu(depth, shared_tt.get_move(board.get_hash()), num_moves_to_unplay);

    for (int i = num_moves_to_unplay - 1; i >= 0; --i)
        board.Board::unplay(moves_to_unplay[i]);
}

void SearchWorker::submit_current_position_to_gpu(int depth, Move tt_move, int ply)
{
    // Assumes `board` is ALREADY at the position to submit -- callers are
    // responsible for getting there (and back) themselves; this just
    // ranks replies with this worker's own heuristic tables (history/
    // killers/continuation history), per gpu_eval::kNumCandidateMoves,
    // same one-shot scorer + partial selection idiom as engine_manager.
    // hpp's root move-scoring path, and pushes the task.
    //
    // The ranking decides WHICH children get CNN-evaluated, so it should
    // match the order the real search will use at this node as closely as
    // possible -- a child the search never visits is a wasted inference.
    // This used to pass score_move(m, 0, 0, 0): no tt_move (so the TT's
    // own best move, by far the likeliest child to be searched, got none
    // of its 9600-point bonus), killers read from ply 0 instead of this
    // node's, and counter-moves disabled. Only ~7.9% of stored scores
    // were ever read back.
    const auto *history = board.get_history();
    const Move prev_move = (history && !history->empty()) ? history->back().move : Move(0);
    // score_move indexes killer_moves[ply]; the PV-leaf caller's walk can
    // in principle reach past the table, so clamp rather than trust it.
    const int safe_ply = std::clamp(ply, 0, engine_constants::search::MaxDepth - 1);

    MoveList list;
    MoveGen::generate_legal_moves(board, list);
    if (board.get_side_to_move() == WHITE)
    {
        for (int i = 0; i < list.size(); ++i)
            list.scores[i] = score_move<WHITE>(list.moves[i], tt_move, safe_ply, prev_move);
    }
    else
    {
        for (int i = 0; i < list.size(); ++i)
            list.scores[i] = score_move<BLACK>(list.moves[i], tt_move, safe_ply, prev_move);
    }

    gpu_eval::GpuTask task;
    task.position = gpu_eval::GpuPosition::from_board(board);
    task.depth = static_cast<std::uint8_t>(std::clamp(depth, 0, 255));
    task.num_candidates = std::min(list.size(), gpu_eval::kNumCandidateMoves);
    for (int i = 0; i < task.num_candidates; ++i)
        task.candidate_moves[i] = list.pick_best_move(i);

    // Phase 0 (voir GpuTask) : la cible, et le choix de l'heuristique privee
    // du coup TT. Re-score les memes candidats avec tt_move supprime, sinon
    // la comparaison serait circulaire.
    if (gpu_eval::measure_diagnostics())
    {
        task.tt_move = tt_move;
        int best_blind = std::numeric_limits<int>::min();
        for (int i = 0; i < task.num_candidates; ++i)
        {
            const Move &m = task.candidate_moves[i];
            const int s = (board.get_side_to_move() == WHITE)
                              ? score_move<WHITE>(m, Move(0), safe_ply, prev_move)
                              : score_move<BLACK>(m, Move(0), safe_ply, prev_move);
            if (s > best_blind)
            {
                best_blind = s;
                task.blind_best = m;
            }
        }
    }

    gpu_eval::shared_gpu_queue().push(task);
}

void SearchWorker::maybe_submit_transposition_to_gpu(int depth, Move tt_move, int ply)
{
    // Called from negamax right after a TT probe that hit -- `board` is
    // already the transposed position, no PV walk needed (unlike
    // maybe_submit_pv_leaf_to_gpu). Depth gating happens at the call
    // site (gpu_eval::kMinDepthForTranspositionSubmit); this only checks
    // the master on/off switch.
    if (!gpu_eval::enabled.load(std::memory_order_relaxed))
        return;
    submit_current_position_to_gpu(depth, tt_move, ply);
}

void SearchWorker::maybe_submit_pv_leaf_to_gpu_throttled(Move pv_root, int depth)
{
    if (depth < gpu_eval::mid_search_submit_min_depth())
        return;
    // Node-count throttle removed (test: does dropping it cause queue
    // overflow / GPU-thread starvation / measurable NPS regression?).
    // GpuQueue::push() still drops silently on a full/contended queue,
    // so this is safe to try -- worst case is more drops, not corruption.
    maybe_submit_pv_leaf_to_gpu(pv_root, depth);
}

int SearchWorker::negamax_with_aspiration(int depth, int last_score)
{
    max_extended_depth = 0;
    int delta = engine_constants::search::aspiration::SmallDelta;
    if (depth >= engine_constants::search::aspiration::HighDepth)
        delta = engine_constants::search::aspiration::HighDelta;
    else if (depth >= engine_constants::search::aspiration::MidDepth)
        delta = engine_constants::search::aspiration::MidDelta;
    int alpha = -engine_constants::eval::Inf;
    int beta = engine_constants::eval::Inf;

    if (depth >= engine_constants::search::aspiration::EnableDepth)
    {
        alpha = last_score - delta;
        beta = last_score + delta;
    }

    int iterations = 0;
    const int max_iterations = engine_constants::search::aspiration::MaxIterations;

    while (true)
    {
        ++iterations;
        int score = negamax(depth, alpha, beta, 0);

        if (abs(score) >= engine_constants::eval::MateScore - depth)
        {
            return score;
        }
        if (abs(score) >= engine_constants::eval::MateScore - engine_constants::search::aspiration::MateWindowMargin)
        {
            alpha = -engine_constants::eval::MateScore;
            beta = engine_constants::eval::MateScore;
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
            delta = std::max(delta * 2, engine_constants::search::aspiration::WidenMinDelta);
            alpha = std::max(-engine_constants::eval::Inf, alpha - delta);
            if (thread_id == 0)
                logs::debug << "info string fail low" << std::endl;
        }
        else if (score >= beta)
        {
            // Fail-high
            delta = std::max(delta * 2, engine_constants::search::aspiration::WidenMinDelta);
            beta = std::min(engine_constants::eval::Inf, beta + delta);
            if (thread_id == 0)
                logs::debug << "info string fail high" << std::endl;
        }

        // Sécurité
        if (iterations >= max_iterations || delta > engine_constants::search::aspiration::WidenMaxDelta)
        {
            alpha = -engine_constants::eval::Inf;
            beta = engine_constants::eval::Inf;
        }
    }
}

void SearchWorker::iterative_deepening()
{
    int last_score = 0;
    for (int depth = 1; depth < engine_constants::search::MaxDepth; ++depth)
    {
        age_history();
        last_score = negamax_with_aspiration(depth, last_score);
        // `board` is guaranteed back at the root here (every play() in
        // negamax is paired with an unplay() on return) -- every
        // worker, not just thread_id == 0 (which only handles UCI
        // "info" reporting below), gets a chance to submit its own PV
        // leaf to the GPU-eval queue.
        maybe_submit_pv_leaf_to_gpu(best_root_move.get_value() != 0 ? best_root_move : out_move, depth);

        // "go depth N" : cette profondeur vient d'etre terminee, on s'arrete.
        // Teste ici, dans la MEME branche que l'arret par le temps, pour que
        // la derniere profondeur complete emette son info line comme
        // d'habitude -- et thread 0 leve shared_stop afin que les autres
        // workers se replient au lieu d'entamer la profondeur suivante.
        const int depth_limit = manager.max_depth_limit();
        const bool depth_reached = depth_limit > 0 && depth >= depth_limit;
        if (depth_reached && thread_id == 0)
            shared_stop.store(true, std::memory_order_relaxed);

        if (depth_reached || shared_stop.load(std::memory_order_relaxed))
        {
            if (thread_id == 0)
            {
                auto elapsed_ms = std::max<long long>(1,
                                                      std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          std::chrono::steady_clock::now() - start_time_ref)
                                                          .count());

                long long nodes = global_nodes.load(std::memory_order_relaxed);
                long long nps = nodes * 1000 / elapsed_ms;
                const Move pv_root = best_root_move.get_value() != 0 ? best_root_move : out_move;
                const std::string pv_line = get_pv_line_with_root(pv_root, depth);
                logs::uci
                    << "info depth " << depth
                    << " seldepth " << max_extended_depth
                    << " score cp " << last_score
                    << " nodes " << nodes
                    << " nps " << nps
                    << " hashfull " << shared_tt.get_hashfull();
                if (!pv_line.empty())
                    logs::uci << " pv " << pv_line;
                logs::uci
                    << std::endl;
            }
            return;
        }
#ifndef NDEBUG
        if (thread_id == 0)
        {
            auto elapsed_ms = std::max<long long>(1,
                                                  std::chrono::duration_cast<std::chrono::milliseconds>(
                                                      std::chrono::steady_clock::now() - start_time_ref)
                                                      .count());

            long long nodes = global_nodes.load(std::memory_order_relaxed);
            long long nps = nodes * 1000 / elapsed_ms;
            logs::debug
                << "info depth " << depth
                << " seldepth " << max_extended_depth
                << " score cp " << last_score
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << shared_tt.get_hashfull()
                << " pv " << get_pv_line(depth)
                << std::endl;
        }
#endif
    }
    if (thread_id == 0)
    {
        shared_stop.store(true, std::memory_order_relaxed);
        auto elapsed_ms = std::max<long long>(1,
                                              std::chrono::duration_cast<std::chrono::milliseconds>(
                                                  std::chrono::steady_clock::now() - start_time_ref)
                                                  .count());

        long long nodes = global_nodes.load(std::memory_order_relaxed);
        long long nps = nodes * 1000 / elapsed_ms;
        const Move pv_root = best_root_move.get_value() != 0 ? best_root_move : out_move;
        const std::string pv_line = get_pv_line_with_root(pv_root, engine_constants::search::MaxDepth - 1);
        logs::uci
            << "info depth " << engine_constants::search::MaxDepth - 1
            << " seldepth " << max_extended_depth
            << " score cp " << last_score
            << " nodes " << nodes
            << " nps " << nps
            << " hashfull " << shared_tt.get_hashfull();
        if (!pv_line.empty())
            logs::uci << " pv " << pv_line;
        logs::uci
            << std::endl;
    }
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
