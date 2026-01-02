#pragma once
#include "engine/engine.hpp"

class EngineManager
{
    Board &main_board;
    TranspositionTable tt;
    double lmr_table[64][64];

    alignas(64) std::atomic<bool> stop_search{false};
    alignas(64) std::atomic<long long> total_nodes{0};

    // Paramètres de temps
    Clock::time_point start_time;
    int time_limit_ms;

public:
    EngineManager(Board &b) : main_board(b)
    {
        tt.resize(2048);
        init_lmr_table();
    }

    void start_search(int time_ms = 20000)
    {
        stop_search = false;
        total_nodes = 0;
        time_limit_ms = time_ms;
        start_time = Clock::now();
        tt.next_generation();

        int num_threads = std::thread::hardware_concurrency();
        std::cout << num_threads << " concurrent threads are supported.\n";

        {
            std::vector<std::jthread> workers;
            for (int i = 0; i < num_threads; ++i)
            {
                workers.emplace_back([this, i]()
                                     {
                SearchWorker worker(main_board, tt, stop_search, total_nodes, start_time, time_limit_ms, lmr_table,i);
                this->run_worker(worker, i); });
            }
        } // Fin du scope : std::jthread appelle join() ici automatiquement

        // --- APPLICATION DU COUP ---
        Move best_move = tt.get_move(main_board.get_hash());
        if (best_move.get_value() != 0)
        {
            std::cout << "bestmove " << best_move.to_uci() << std::endl;
            main_board.play(best_move);
        }
        else
        {
            // Fallback si la recherche a été interrompue trop tôt
            std::cout << "info string search failed to find a move in TT" << std::endl;
        }
    }

    int evaluate_position(int time_ms)
    {
        // 1. Réinitialisation des flags
        stop_search = false;
        total_nodes = 0;
        time_limit_ms = time_ms;
        start_time = Clock::now();

        // 2. Création d'un worker unique (pas besoin de multithread pour un simple eval)
        SearchWorker worker(main_board, tt, stop_search, total_nodes, start_time, time_limit_ms, lmr_table, 0);

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

private:
    void run_worker(SearchWorker &worker, int thread_id)
    {
        int last_score = 0;

        for (int current_depth = 1; current_depth <= MAX_DEPTH; ++current_depth)
        {
            if (stop_search.load(std::memory_order_relaxed))
                break;

            // --- DIVERSIFICATION LAZY SMP ---
            // Le thread 0 suit les profondeurs normalement.
            // Les helpers explorent parfois des profondeurs décalées pour aider la TT.
            int search_depth = current_depth;
            if (thread_id > 0)
            {
                // Exemple : threads impairs cherchent à depth + 1
                search_depth = (thread_id % 2 == 0) ? current_depth : current_depth + 1;
            }

            int score = worker.negamax_with_aspiration(search_depth, last_score);

            if (stop_search.load(std::memory_order_relaxed))
                break;

            // --- SEUL LE THREAD MAÎTRE COMMUNIQUE ---
            if (thread_id == 0)
            {
                last_score = score;
                auto now = Clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();

                long long nodes = total_nodes.load(std::memory_order_relaxed);
                double nps = elapsed > 0 ? (nodes * 1000.0 / elapsed) : 0;

                std::cout << "info depth " << current_depth
                          << " score cp " << score
                          << " nodes " << nodes
                          << " nps " << (long long)nps
                          << " hashfull " << tt.get_hashfull()
                          << " pv " << worker.get_pv_line(current_depth)
                          << std::endl;

                // Arrêt si Mat trouvé
                if (std::abs(score) >= MATE_SCORE - MAX_DEPTH)
                {
                    stop_search.store(true, std::memory_order_relaxed);
                    break;
                }
            }
        }
    }
    void init_lmr_table()
    {
        for (int d = 1; d < 64; d++)
        {
            for (int m = 1; m < 64; m++)
            {
                // Formule standard  K = 2.25
                lmr_table[d][m] = 0.5 + log(d) * log(m) / 2.25;
            }
        }
    }
};