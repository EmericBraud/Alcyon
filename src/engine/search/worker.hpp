#pragma once

#include <algorithm>
#include <cstdlib>
#include <chrono>

#include <atomic>

#include "engine/eval/tablebase.hpp"
#include "engine/config/config.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/tt/transp_table.hpp"
#include "engine/eval/virtual_board.hpp"

class EngineManager;

// Diagnostic (CHESS26_TT_NO_CUTOFF=1) : la TT ne fournit plus que des COUPS,
// plus aucune coupure de score -- ni dans negamax ni dans qsearch. Sert a
// borner le gain maximal qu'un ordonnancement parfait pourrait rapporter : on
// compare le nombre de noeuds d'une recherche a TT vide a celui d'une
// re-recherche sur une TT deja remplie par une recherche de meme profondeur.
// Sans ce drapeau la seconde recherche gagnerait surtout par ses coupures, ce
// qui surestimerait massivement l'apport de l'ordonnancement.
namespace search
{
    // Diagnostic (CHESS26_ORDER_STATS=1, affiche par la commande UCI
    // "orderstats") : a quel RANG se trouve le coup qui provoque un
    // fail-high, et est-il tactique ou calme ?
    //
    // C'est la mesure qui dimensionne l'idee de reordonner la QUEUE de liste
    // (quiets et mauvaises captures) avec une tete policy : si la quasi-
    // totalite des coupures tombe sur les deux premiers coups, cette queue
    // est rarement atteinte et la reordonner ne peut rien rapporter. Si une
    // part notable tombe au rang 5 ou plus, il y a de la place.
    //
    // Buckets : rang 1, 2, 3, 4, 5-8, 9-16, 17+.
    constexpr int kCutoffBuckets = 7;
    inline std::atomic<long long> cutoff_tactical[kCutoffBuckets] = {};
    inline std::atomic<long long> cutoff_quiet[kCutoffBuckets] = {};

    // CHESS26_SEARCH_EXPERIMENTS (option CMake, off par defaut) : ces deux
    // interrupteurs n'existent que pour les mesures ci-dessus et pour
    // l'experience d'oracle d'ordonnancement. Ils etaient lus par
    // getenv une fois, mais la fonction restait un appel avec son garde
    // d'initialisation de static local -- teste a CHAQUE noeud pour
    // tt_cutoffs_enabled(). Hors build d'experimentation ils deviennent des
    // constantes, donc les conditions qui les portent disparaissent.
#ifdef CHESS26_SEARCH_EXPERIMENTS
    inline bool order_stats_enabled()
    {
        static const bool on = std::getenv("CHESS26_ORDER_STATS") != nullptr;
        return on;
    }
#else
    constexpr bool order_stats_enabled() { return false; }
#endif

    inline int cutoff_bucket(int rank)
    {
        if (rank <= 4) return rank - 1;
        if (rank <= 8) return 4;
        if (rank <= 16) return 5;
        return 6;
    }

    inline void record_cutoff(int rank, bool is_tactical)
    {
        const int b = cutoff_bucket(rank);
        (is_tactical ? cutoff_tactical : cutoff_quiet)[b].fetch_add(1, std::memory_order_relaxed);
    }

#ifdef CHESS26_SEARCH_EXPERIMENTS
    inline bool tt_cutoffs_enabled()
    {
        static const bool on = std::getenv("CHESS26_TT_NO_CUTOFF") == nullptr;
        return on;
    }
#else
    constexpr bool tt_cutoffs_enabled() { return true; }
#endif
}

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
    Move killer_moves[engine_constants::search::MaxDepth][2];
    Move counter_moves[2][7][64];
    int continuation_hist_1[2][7][64][64]; // [side][piece][from][to] for 1-ply continuation
    int continuation_hist_2[2][7][64][64]; // [side][piece][from][to] for 2-ply continuation
    std::array<Move, engine_constants::search::MaxDepth> move_stack;

    // Pile de recherche : une evaluation statique par ply.
    //
    // Deux roles, et le second est celui qui justifie la structure :
    //  1. CACHE. Razoring, RFP et futility appelaient chacun
    //     Eval::prune_eval_relative sur la MEME position -- soit jusqu'a
    //     trois passes reseau identiques par noeud, puisque eval() ignore
    //     la fenetre alpha/beta qu'on lui passe (voir
    //     nnue/pos_eval.cpp : elle ne sert plus depuis la suppression de
    //     l'eval paresseuse). Le remplissage est PARESSEUX : un noeud qui
    //     n'evaluait pas n'evalue toujours pas, donc l'operation ne peut
    //     qu'enlever des evals, jamais en ajouter.
    //  2. improving. Comparer l'eval de ce noeud a celle de l'ancetre
    //     ply-2 (meme camp au trait) dit si notre position s'ameliore.
    //     Impossible sans garder les evals des ancetres.
    //
    // kEvalNone marque "pas encore calculee" : hors de portee d'un score
    // reel, tous bornes par eval::Inf.
    static constexpr int kEvalNone = 1 << 30;
    int static_eval_stack[engine_constants::search::MaxDepth + 8];

    // Métriques locales
    long long local_nodes = 0;
    int thread_id;

    Move best_root_move = 0;
    Move out_move = 0;

    int max_extended_depth;

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
    // cut_node : PREDICTION du type de noeud au sens de Knuth-Moore, portee
    // par la recursion PVS (voir les sites d'appel dans negamax.cpp). Elle a
    // le droit de se tromper : elle ne pilote que des heuristiques, jamais un
    // score ni une borne. Un noeud a fenetre nulle est cut ou all, et ce
    // booleen est exactement le bit qui les distingue.
    int negamax(int depth, int alpha, int beta, int ply, bool allow_null, bool cut_node, Move excluded_move = 0);
    inline int negamax(int depth, int alpha, int beta, int ply)
    {
        if (board.get_side_to_move() == WHITE)
        {
            return negamax<WHITE>(depth, alpha, beta, ply, true, false);
        }
        return negamax<BLACK>(depth, alpha, beta, ply, true, false);
    }

    template <Color Us>
    int qsearch(int alpha, int beta, int ply);

    // --- Heuristiques ---
    void clear_heuristics()
    {
        std::memset(history_moves, 0, sizeof(history_moves));
        std::memset(killer_moves, 0, sizeof(killer_moves));
        std::memset(counter_moves, 0, sizeof(counter_moves));
        std::memset(continuation_hist_1, 0, sizeof(continuation_hist_1));
        std::memset(continuation_hist_2, 0, sizeof(continuation_hist_2));
        for (int i = 0; i < engine_constants::search::MaxDepth + 8; ++i)
            static_eval_stack[i] = kEvalNone;
    }

    // Gravite : l'entree est attiree vers 0 proportionnellement a sa valeur,
    // donc bornee dans +/-HistMax sans clamp et auto-decroissante. Remplace
    // l'ancien couple (bonus non borne, malus clampe) qui saturait les tables
    // vers le haut et figeait l'ordonnancement en milieu de partie -- et du
    // meme coup age_history() et son /= 8.
    static void update_hist(int &entry, int bonus)
    {
        const int max = engine_constants::search::move_ordering::HistMax;
        bonus = std::clamp(bonus, -max, max);
        entry += bonus - entry * std::abs(bonus) / max;
    }

    // --- utilitaires ---
    template <Color Us>
    int score_move(const Move &move, const Move &tt_move, int ply, const Move &prev_move) const;
    int score_capture(const Move &move) const;
    int score_quiet_history(int raw_score, const Move &move, const Move &prev_move, const Move &prev_prev_move, Color us) const;
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