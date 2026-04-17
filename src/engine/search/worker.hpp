#pragma once
#include <chrono>

#include <atomic>

#include "engine/eval/tablebase.hpp"
#include "engine/config/config.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/tt/transp_table.hpp"
#include "engine/eval/virtual_board.hpp"

class EngineManager;

struct SearchWorker
{
    const EngineManager &manager;
    // Ressources locales (Copie pour éviter les Data Races)
    VBoard board;

    // Ressources partagées (Références vers l'Orchestrateur)
    TranspositionTable &shared_tt;
    TableBase &shared_tb;
    std::atomic<bool> &shared_stop;
    std::atomic<long long> &global_nodes;
    const std::chrono::steady_clock::time_point start_time_ref;
    const int time_limit_ms_ref;
    const double (&lmr_table)[64][64];

    // Heuristiques locales (Thread-local)
    int history_moves[2][64][64];
    int continuation_hist_1[2][7][64][64];
    int continuation_hist_2[2][7][64][64];
    Move killer_moves[engine_constants::search::MaxDepth][2];
    Move counter_moves[2][7][64];
    std::array<Move, engine_constants::search::MaxDepth> move_stack;

    // Métriques locales
    long long local_nodes = 0;
    int thread_id;

    Move best_root_move = 0;
    Move out_move = 0;

    int max_extended_depth;
    std::array<int, engine_constants::search::MaxDepth> static_eval_stack{};
    std::array<bool, engine_constants::search::MaxDepth> improving_stack{};
    std::array<bool, engine_constants::search::MaxDepth> pv_stack{};

    // CONSTRUCTEUR PRINCIPAL
    // Appelé par l'orchestrateur pour chaque thread
    SearchWorker(
        const EngineManager &e,
        const VBoard &b,
        TranspositionTable &tt,
        TableBase &tb,
        std::atomic<bool> &stop,
        std::atomic<long long> &nodes,
        const std::chrono::steady_clock::time_point &start_time,
        const int &time_limit,
        const double (&lmr)[64][64],
        int id)
        : manager(e),
          board(b), // Copie physique du plateau
          shared_tt(tt),
          shared_tb(tb),
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
    int negamax(int depth, int alpha, int beta, int ply, bool allow_null, Move excluded_move = 0);
    inline int negamax(int depth, int alpha, int beta, int ply)
    {
        if (board.get_side_to_move() == WHITE)
        {
            return negamax<WHITE>(depth, alpha, beta, ply, true);
        }
        return negamax<BLACK>(depth, alpha, beta, ply, true);
    }

    template <Color Us>
    int qsearch(int alpha, int beta, int ply);

    // --- Heuristiques ---
    void clear_heuristics()
    {
        std::memset(history_moves, 0, sizeof(history_moves));
        std::memset(continuation_hist_1, 0, sizeof(continuation_hist_1));
        std::memset(continuation_hist_2, 0, sizeof(continuation_hist_2));
        std::memset(killer_moves, 0, sizeof(killer_moves));
        std::memset(counter_moves, 0, sizeof(counter_moves));
    }

    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                {
                    history_moves[c][f][t] /= 8;
                    for (int p = 0; p < 7; ++p)
                    {
                        continuation_hist_1[c][p][f][t] /= 8;
                        continuation_hist_2[c][p][f][t] /= 8;
                    }
                }
    }

    static constexpr int HistoryLimit = 12000;

    inline void update_history_entry(int &entry, int bonus)
    {
        bonus = std::clamp(bonus, -HistoryLimit, HistoryLimit);
        entry += bonus - entry * std::abs(bonus) / HistoryLimit;
        entry = std::clamp(entry, -HistoryLimit, HistoryLimit);
    }

    template <Color Us>
    inline void update_quiet_histories(Move m, Move prev_move, Move prev_prev_move, int bonus)
    {
        update_history_entry(history_moves[Us][m.get_from_sq()][m.get_to_sq()], bonus);

        const int to = m.get_to_sq();
        if (prev_move != 0)
        {
            const int prev_piece = prev_move.get_from_piece();
            const int prev_to = prev_move.get_to_sq();
            update_history_entry(continuation_hist_1[Us][prev_piece][prev_to][to], bonus);
        }
        if (prev_prev_move != 0)
        {
            const int prev2_piece = prev_prev_move.get_from_piece();
            const int prev2_to = prev_prev_move.get_to_sq();
            update_history_entry(continuation_hist_2[Us][prev2_piece][prev2_to][to], bonus);
        }
    }

    // --- utilitaires ---
    template <Color Us>
    int score_move(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;
    int score_capture(const Move &move) const;
    int score_quiet_history(int raw_score, const Move &move, int ply, const Move &prev_move, const Move &prev_prev_move) const;
    template <Color Side>
    int see(int sq, Piece target, Piece attacker, int from_sq) const;
    std::string get_pv_line(int depth);
    std::string get_pv_line_with_root(Move root_move, int depth);
    int negamax_with_aspiration(int depth, int last_score);

    inline VBoard &get_board()
    {
        return board;
    }

    void iterative_deepening();

    TranspositionTable &get_tt()
    {
        return shared_tt;
    }
    bool check_stop();
};