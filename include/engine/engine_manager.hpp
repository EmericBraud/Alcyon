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

    // Paramètres de temps
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::atomic<int> time_limit{0};
    std::atomic<Move> root_best_move;

    std::atomic<int> current_depth{0};
    std::atomic<int> next_root_index{0};
    std::atomic<bool> depth_finished{false};
    std::atomic<int> finished_root_moves{0};

    std::mutex root_mutex;
    Move depth_best_move;
    int depth_best_score;

public:
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

    void start_search(int time_ms = 20000, bool ponder = false, bool infinite = false)
    {
        stop(); // On arrête une recherche précédente si elle existe
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
                                     { this->iterative_deepening(); });
    }

    bool should_stop() const
    {
        if (stop_search.load(std::memory_order_relaxed))
            return true;

        if (!is_pondering.load())
        {
            if (!is_infinite.load())
            {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count();
                return elapsed >= time_limit.load();
            }
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
        for (int d = 1; d < 64; d++)
        {
            for (int m = 1; m < 64; m++)
            {
                // Formule standard  K = 2.25
                lmr_table[d][m] = 0.5 + log(d) * log(m) / 2.25;
            }
        }
    }
    void iterative_deepening()
    {
        // =========================
        // Génération des coups root
        // =========================
        MoveList moves;
        MoveGen::generate_legal_moves(main_board, moves);

        if (moves.empty())
        {
            std::cout << "bestmove 0000\n";
            return;
        }

        std::vector<RootMove> root_moves;
        root_moves.reserve(moves.size());
        for (Move m : moves)
            root_moves.emplace_back(m);

        const int num_threads =
            std::max(1u, std::thread::hardware_concurrency());

        const int root_sign =
            (main_board.get_side_to_move() == WHITE) ? +1 : -1;

        Move best_move = root_moves[0].move;
        int best_score = -INF;

        // =========================
        // Création des workers
        // =========================
        std::vector<SearchWorker> workers;
        workers.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t)
        {
            workers.emplace_back(
                *this,
                main_board,
                tt,
                stop_search,
                total_nodes,
                start_time,
                time_limit,
                lmr_table,
                t);
        }

        // =========================
        // Variables SMP
        // =========================
        std::atomic<int> current_depth{0};
        std::atomic<int> next_root_index{0};
        std::atomic<int> finished_root_moves{0};

        // =========================
        // Boucle worker (Lazy SMP)
        // =========================
        auto worker_fn = [&](int tid)
        {
            SearchWorker &worker = workers[tid];
            Board &board = worker.get_board();

            int local_depth = 0;

            while (!stop_search.load(std::memory_order_relaxed))
            {
                int d = current_depth.load(std::memory_order_acquire);
                if (d == 0 || d == local_depth)
                {
                    std::this_thread::yield();
                    continue;
                }

                local_depth = d;

                while (true)
                {
                    int idx = next_root_index.fetch_add(
                        1, std::memory_order_relaxed);

                    if (idx >= (int)root_moves.size())
                        break;

                    RootMove &rm = root_moves[idx];
                    Move m = rm.move;

                    board.play(m);

                    int score;
                    if (d > 5)
                        score = -worker.negamax_with_aspiration(
                            d - 1, -rm.score);
                    else
                        score = -worker.negamax(d - 1, -INF, INF, 0);

                    board.unplay(m);

                    rm.score = score;
                    finished_root_moves.fetch_add(
                        1, std::memory_order_relaxed);
                }
            }
        };

        // =========================
        // Lancement des threads
        // =========================
        std::vector<std::jthread> threads;
        threads.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t)
            threads.emplace_back(worker_fn, t);

        // =========================
        // ITERATIVE DEEPENING
        // =========================
        for (int depth = 1; depth <= MAX_DEPTH; ++depth)
        {
            if (should_stop())
                break;

            // Root ordering (PV first)
            std::sort(root_moves.begin(), root_moves.end(),
                      [](const RootMove &a, const RootMove &b)
                      {
                          return a.score > b.score;
                      });

            next_root_index.store(0, std::memory_order_relaxed);
            finished_root_moves.store(0, std::memory_order_relaxed);
            current_depth.store(depth, std::memory_order_release);

            // Attente fin de profondeur
            while (finished_root_moves.load(
                       std::memory_order_relaxed) <
                   (int)root_moves.size())
            {
                if (should_stop())
                    break;
                std::this_thread::sleep_for(
                    std::chrono::microseconds(50));
            }

            // Sélection du meilleur coup
            best_score = -INF;
            for (const RootMove &rm : root_moves)
            {
                if (rm.score > best_score)
                {
                    best_score = rm.score;
                    best_move = rm.move;
                }
            }

            // =========================
            // INFO UCI
            // =========================
            auto elapsed_ms =
                std::max<long long>(
                    1,
                    std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() -
                        start_time)
                        .count());

            long long nodes =
                total_nodes.load(std::memory_order_relaxed);
            long long nps = nodes * 1000 / elapsed_ms;

            std::cout
                << "info depth " << depth
                << " score cp " << (root_sign * best_score)
                << " nodes " << nodes
                << " nps " << nps
                << " hashfull " << tt.get_hashfull()
                << " pv " << best_move.to_uci()
                << std::endl;
        }

        root_best_move.store(best_move, std::memory_order_relaxed);
        std::cout << "bestmove " << best_move.to_uci() << std::endl;
    }
};