#pragma once
#include "engine/engine.hpp"

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

    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::atomic<int> time_limit{0};
    std::atomic<Move> root_best_move;

    std::mutex root_mutex;
    Move depth_best_move;
    int depth_best_score;

public:
    inline Move get_root_best_move() const
    {
        return root_best_move;
    }
    EngineManager(Board &b) : main_board(b)
    {
        tt.resize(512);
        init_lmr_table();
        root_best_move = 0;
    }

    void stop()
    {
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

    void start_search(int time_ms = 20000, bool ponder = false, bool infinite = false, bool play_move = false)
    {
        stop();
        if (search_thread.joinable())
        {
            stop_search.store(true);
            search_thread.join();
        }

        stop_search.store(false, std::memory_order_relaxed);
        is_pondering.store(ponder, std::memory_order_relaxed);
        is_infinite.store(infinite, std::memory_order_relaxed);
        total_nodes.store(0, std::memory_order_relaxed);
        root_best_move.store(0);

        time_limit.store(time_ms, std::memory_order_relaxed);
        start_time = std::chrono::steady_clock::now();
        search_thread = std::jthread([this]()
                                     { this->start_workers(); });
        search_thread.join();
        if (play_move)
        {
            main_board.play(root_best_move);
        }
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
        start_time = Clock::now();
        tt.next_generation();

        // 2. Création d'un worker unique (pas besoin de multithread pour un simple eval)
        SearchWorker worker(*this, main_board, tt, stop_search, total_nodes, start_time, time_limit, lmr_table, 0);

        // 3. Recherche par itérations successives (Iterative Deepening)
        int score = 0;
        for (int d = 1; d <= MAX_DEPTH; ++d)
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
        is_pondering = false;
        start_time = std::chrono::steady_clock::now();
    }

private:
    void init_lmr_table()
    {
        for (int d = 1; d < 64; ++d)
            for (int m = 1; m < 64; ++m)
                lmr_table[d][m] = 0.5 + log(d) * log(m) / 2.25;
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

        if (best_move.get_value() == 0)
        {
            // On essaie de trouver n'importe quel coup légal pour ne pas crash
            MoveList list;
            MoveGen::generate_legal_moves(main_board, list); // Suppose que tu as cette fonction ou similaire

            if (list.count > 0)
            {
                // Panic mode : on joue le premier coup légal trouvé
                best_move = list.moves[0];
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
        logs::uci << "bestmove " << best_move.to_uci() << std::endl;
    }
};
