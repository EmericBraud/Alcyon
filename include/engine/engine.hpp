#pragma once
#include "engine/eval/pos_eval.hpp"

#define MAX_DEPTH 50
constexpr int INF = 10000;

class EngineManager;

using Clock = std::chrono::steady_clock;

class SearchWorker
{
private:
    const EngineManager &manager;
    // Ressources locales (Copie pour éviter les Data Races)
    Board board;

    // Ressources partagées (Références vers l'Orchestrateur)
    TranspositionTable &shared_tt;
    std::atomic<bool> &shared_stop;
    std::atomic<long long> &global_nodes;
    const Clock::time_point start_time_ref;
    const int time_limit_ms_ref;
    const double (&lmr_table)[64][64];

    // Heuristiques locales (Thread-local)
    int history_moves[2][64][64];
    Move killer_moves[MAX_DEPTH][2];
    Move counter_moves[2][7][64];
    std::array<Move, MAX_DEPTH> move_stack;

    // Métriques locales
    long long local_nodes = 0;
    int thread_id;

public:
    Move best_root_move = 0;

    // CONSTRUCTEUR PRINCIPAL
    // Appelé par l'orchestrateur pour chaque thread
    SearchWorker(
        const EngineManager &e,
        const Board &b,
        TranspositionTable &tt,
        std::atomic<bool> &stop,
        std::atomic<long long> &nodes,
        const Clock::time_point &start_time,
        const int &time_limit,
        const double (&lmr)[64][64],
        int id)
        : manager(e),
          board(b), // Copie physique du plateau
          shared_tt(tt),
          shared_stop(stop),
          global_nodes(nodes),
          start_time_ref(start_time),
          time_limit_ms_ref(time_limit),
          lmr_table(lmr),
          thread_id(id)
    {
        clear_heuristics();
    }

    // --- Méthodes de recherche ---
    template <Color Us>
    int negamax(int depth, int alpha, int beta, int ply);
    inline int negamax(int depth, int alpha, int beta, int ply)
    {
        if (board.get_side_to_move() == WHITE)
        {
            return negamax<WHITE>(depth, alpha, beta, ply);
        }
        return negamax<BLACK>(depth, alpha, beta, ply);
    }

    template <Color Us>
    int qsearch(int alpha, int beta, int ply);

    // --- Heuristiques ---
    void clear_heuristics()
    {
        std::memset(history_moves, 0, sizeof(history_moves));
        std::memset(killer_moves, 0, sizeof(killer_moves));
        std::memset(counter_moves, 0, sizeof(counter_moves));
    }

    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                    history_moves[c][f][t] /= 8;
    }

    // --- utilitaires ---
    int score_move(const Move &move, const Board &current_board, const Move &tt_move, int ply, const Move &prev_move) const;
    int score_capture(const Move &move) const;
    int see(int sq, Piece target, Piece attacker, Color side, int from_sq) const;
    std::string get_pv_line(int depth);
    int negamax_with_aspiration(int depth, int last_score);

    inline Board &get_board()
    {
        return board;
    }
};