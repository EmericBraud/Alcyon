#pragma once
#include "engine/pos_eval.hpp"

#define MAX_DEPTH 50
constexpr int INF = 1000000;

struct MoveScorer
{
    Move m;
    int score;
};

using Clock = std::chrono::steady_clock;

class Engine
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

    int qsearch(int alpha, int beta);
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

    int see(int sq, Piece target, Piece attacker, Color side, int from_sq) const;

public:
    int negamax(int depth, int alpha, int beta, int ply);
    void play(int time_ms = 2000);

    int score_move(const Move &move, const Board &board, const Move &tt_move, int ply) const;
    // Score uniquement pour le Quiescence Search (MVV-LVA)
    int score_capture(const Move &move) const;
    void age_history()
    {
        for (int c = 0; c < 2; ++c)
            for (int f = 0; f < 64; ++f)
                for (int t = 0; t < 64; ++t)
                    history_moves[c][f][t] /= 8;
    }

    std::string get_pv_line(int depth);

public:
    Engine(Board &board) : board(board)
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