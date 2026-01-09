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
    Move root_best_move;

public:
    EngineManager(Board &b) : main_board(b)
    {
        tt.resize(64);
        init_lmr_table();
        root_best_move = 0;
    }

    void clear()
    {
        tt.clear();
        stop_search.store(false);
        total_nodes.store(0);
    }

    void start_search(int time_ms = 5000)
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

        if (root_best_move.get_value() != 0)
        {
            std::cout << "bestmove " << root_best_move.to_uci() << std::endl;
            main_board.play(root_best_move);
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
            // 1. Vérification d'arrêt avant de lancer une nouvelle itération
            if (stop_search.load(std::memory_order_relaxed))
                break;

            // 2. Diversification Lazy SMP
            int search_depth = current_depth;
            if (thread_id > 0)
            {
                // Décalage pour que les helpers explorent des branches différentes
                // On ajoute un offset basé sur l'ID pour maximiser la couverture TT
                search_depth = current_depth + (thread_id % 4);
            }

            // Optionnel : On ne réduit l'historique que sur le thread maître pour garder les heuristiques helpers
            if (thread_id == 0)
                worker.age_history();

            // 3. Lancement de la recherche
            int score = worker.negamax_with_aspiration(search_depth, last_score);

            // 4. Vérification d'arrêt post-recherche
            // Si le temps a expiré PENDANT negamax, le score et le coup sont potentiellement corrompus
            if (stop_search.load(std::memory_order_relaxed))
                break;

            // 5. Logique spécifique au Thread Maître (ID 0)
            if (thread_id == 0)
            {
                last_score = score; // Mise à jour pour la prochaine fenêtre d'aspiration

                Move current_best = tt.get_move(main_board.get_hash());
                if (current_best.get_value() != 0)
                {
                    // On stocke ce coup dans une variable membre de EngineManager
                    this->root_best_move = current_best;
                }

                // --- AFFICHAGE UCI ---
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