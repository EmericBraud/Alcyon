#pragma once
#include <atomic>
#include <chrono>
#include <thread>
#include <cmath>
#include <experimental/scope>

#include "common/logger.hpp"
#include "core/move/move.hpp"
#include "core/board/board.hpp"
#include "core/move/generator/move_generator.hpp"

#include "engine/config/config.hpp"
#include "engine/tt/transp_table.hpp"
#include "engine/search/worker.hpp"

struct RootMove
{
    Move move;
    int score;

    RootMove(Move m) : move(m), score(0) {}
};

class EngineManager
{
    Board &main_board;
    TranspositionTable tt;
    double lmr_table[64][64];

    std::jthread search_thread;
    alignas(64) std::atomic<bool> stop_search{false};
    alignas(64) std::atomic<bool> is_pondering{false};
    alignas(64) std::atomic<long long> total_nodes{0};
    alignas(64) std::atomic<bool> is_infinite{false};
    alignas(64) std::atomic<bool> ponder_enabled{false};

    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::atomic<int> time_limit{0};
    std::atomic<Move> root_best_move;

    Move depth_best_move;
    int depth_best_score;

public:
    inline Move get_root_best_move() const
    {
        return root_best_move.load(std::memory_order_relaxed);
    }
    EngineManager(Board &b) : main_board(b)
    {
        tt.resize(512);
        init_lmr_table();
        root_best_move = 0;
    }

    void stop()
    {
        is_pondering.store(false, std::memory_order_relaxed);
        stop_search.store(true, std::memory_order_relaxed);
    }

    void clear()
    {
        tt.clear();
        stop_search.store(false);
        total_nodes.store(0);
        is_pondering.store(false);
        root_best_move.store(0);
    }

    void start_search(int time_ms = 20000, bool ponder = false, bool infinite = false, bool ponder_enabled = false)
    {
        stop();
        if (search_thread.joinable())
        {
            stop_search.store(true);
            search_thread.join();
        }
        tt.next_generation();

        stop_search.store(false, std::memory_order_relaxed);
        is_pondering.store(ponder, std::memory_order_relaxed);
        is_infinite.store(infinite, std::memory_order_relaxed);
        this->ponder_enabled.store(ponder_enabled, std::memory_order_relaxed);
        total_nodes.store(0, std::memory_order_relaxed);
        root_best_move.store(0);

        time_limit.store(time_ms, std::memory_order_relaxed);
        start_time = std::chrono::steady_clock::now();
        search_thread = std::jthread([this]()
                                     { this->start_workers(); });
    }

    bool should_stop() const
    {
        if (stop_search.load(std::memory_order_relaxed))
            return true;

        if (!is_pondering.load() && !is_infinite.load())
        {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
            return elapsed >= time_limit.load();
        }
        return false;
    }

    int evaluate_position(int time_ms)
    {
        // 1. Réinitialisation des flags
        stop_search = false;
        total_nodes = 0;
        time_limit = time_ms;
        start_time = std::chrono::steady_clock::now();
        tt.next_generation();

        // 2. Création d'un worker unique (pas besoin de multithread pour un simple eval)
        SearchWorker worker(*this, main_board, tt, stop_search, total_nodes, start_time, time_limit, lmr_table, 0);

        // 3. Recherche par itérations successives (Iterative Deepening)
        int score = 0;
        for (int d = 1; d <= engine::config::search::MaxDepth; ++d)
        {
            score = worker.negamax_with_aspiration(d, score);

            // Arrêt prématuré si temps écoulé
            if (stop_search.load(std::memory_order_relaxed))
                break;
        }

        return score;
    }

    void convert_ponder_to_real()
    {
        is_pondering.store(false, std::memory_order_relaxed);
        start_time = std::chrono::steady_clock::now();
    }

    inline TranspositionTable &get_tt()
    {
        return tt;
    }

    void wait()
    {
        if (search_thread.joinable())
        {
            search_thread.join();
        }
    }

private:
    void init_lmr_table()
    {
        for (int d = 1; d < 64; ++d)
            for (int m = 1; m < 64; ++m)
                lmr_table[d][m] = 0.5 + std::log(d) * std::log(m) / 2.25;
    }

    void start_workers()
    {

        const int num_threads = std::max(1u, std::thread::hardware_concurrency());
        Move best_move;

        // =========================
        // Création des workers
        // =========================
        std::vector<SearchWorker> workers;
        workers.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t)
            workers.emplace_back(*this, main_board, tt, stop_search, total_nodes, start_time, time_limit, lmr_table, t);

        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (auto &worker : workers)
            threads.emplace_back(&SearchWorker::iterative_deepening, &worker);

        for (auto &th : threads)
            th.join();

        best_move = workers[0].best_root_move;

        if (best_move.get_value() == 0) [[unlikely]]
        {
            logs::uci << "PANICK MODE" << std::endl;
            // Panick mode : we try to find the best possible legal move
            // First attempt : transp table
            best_move = tt.get_move(main_board.get_hash());
            if (main_board.is_move_pseudo_legal(best_move) && main_board.is_move_legal(best_move))
            {
                logs::uci << "bestmove " << best_move.to_uci() << std::endl;
                return;
            }

            // Second attempt : we pick the most promising move
            MoveList list;
            MoveGen::generate_legal_moves(main_board, list);
            for (int i = 0; i < list.size(); ++i)
                list.scores[i] = workers[0].score_move(list.moves[i], main_board, 0, 0, 0);

            if (list.count > 0) [[likely]]
            {
                // Panic mode : on joue le coup le plus prometteur
                best_move = list.pick_best_move(0);
                logs::debug << "info string WARNING: Search returned 0, playing emergency move." << std::endl;
            }
            else
            {
                // Vraiment aucun coup (Mat ou Pat)
                // UCI requiert "bestmove (none)" dans certains cas, ou juste null
                logs::uci << "bestmove (none)" << std::endl;
                root_best_move.store(0, std::memory_order_relaxed);
                return;
            }
        }

        // =========================
        root_best_move.store(best_move, std::memory_order_relaxed);

        if (is_pondering.load(std::memory_order_relaxed))
            return;

        if (ponder_enabled.load(std::memory_order_relaxed))
        {
            main_board.play(best_move);
            auto guard = std::experimental::scope_exit([&best_move, this]
                                                       { main_board.unplay(best_move); });
            Move second_move = tt.get_move(main_board.get_hash());
            if (main_board.is_move_pseudo_legal(second_move) && main_board.is_move_legal(second_move))
            {
                logs::uci << "bestmove " << best_move.to_uci() << " ponder " << second_move.to_uci() << std::endl;
                return;
            }
        }
        logs::uci << "bestmove " << best_move.to_uci() << std::endl;
    }
};
