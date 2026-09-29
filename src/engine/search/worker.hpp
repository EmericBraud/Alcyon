#pragma once

#include <algorithm>
#include <string>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>

#include <atomic>

#include "engine/eval/tablebase.hpp"
#include "engine/config/config.hpp"
#include "engine/eval/pos_eval.hpp"
#include "engine/tt/transp_table.hpp"
#include "engine/eval/virtual_board.hpp"
#include "engine/utils/random.hpp"

class EngineManager;

// Diagnostic (ALCYON_TT_NO_CUTOFF=1) : la TT ne fournit plus que des COUPS,
// plus aucune coupure de score -- ni dans negamax ni dans qsearch. Sert a
// borner le gain maximal qu'un ordonnancement parfait pourrait rapporter : on
// compare le nombre de noeuds d'une recherche a TT vide a celui d'une
// re-recherche sur une TT deja remplie par une recherche de meme profondeur.
// Sans ce drapeau la seconde recherche gagnerait surtout par ses coupures, ce
// qui surestimerait massivement l'apport de l'ordonnancement.
namespace search
{
    // Diagnostic (ALCYON_ORDER_STATS=1, affiche par la commande UCI
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

    // ALCYON_SEARCH_EXPERIMENTS (option CMake, off par defaut) : ces deux
    // interrupteurs n'existent que pour les mesures ci-dessus et pour
    // l'experience d'oracle d'ordonnancement. Ils etaient lus par
    // getenv une fois, mais la fonction restait un appel avec son garde
    // d'initialisation de static local -- teste a CHAQUE noeud pour
    // tt_cutoffs_enabled(). Hors build d'experimentation ils deviennent des
    // constantes, donc les conditions qui les portent disparaissent.
#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline bool order_stats_enabled()
    {
        static const bool on = std::getenv("ALCYON_ORDER_STATS") != nullptr;
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

    // Diagnostic (ALCYON_PRUNE_STATS=1, affiche par la commande UCI
    // "prunestats") : etape 0 de docs/learned-pruning.md. Pour chaque noeud
    // non-PV, hors echec, ply > 0, depth >= 1 : comment il se termine, et,
    // s'il est cherche, a quel point (static_eval - beta) predisait deja
    // son resultat. C'est la marge de manoeuvre d'un reseau "ce noeud vaut-il
    // d'etre calcule" : si les noeuds cherches sont deja imprevisibles avec
    // la seule eval, le reseau doit trouver l'information ailleurs.
    //
    // Sous-arbre : differences de global_nodes + local_nodes, donc exact en
    // mono-thread seulement (bench a Threads=1).
    constexpr int kPruneDepths = 10; // depth 1..9, puis 10+
    enum PruneOutcome { PO_RAZOR, PO_RFP, PO_NMP, PO_FAIL_HIGH, PO_FAIL_LOW, PO_LEARNED, kPruneOutcomes };
    // Bornes des buckets de (static_eval - beta), en cp.
    constexpr int kMarginEdges[] = {-400, -200, -100, -50, 0, 50, 100, 200, 400};
    constexpr int kMarginBuckets = sizeof(kMarginEdges) / sizeof(int) + 1;
    inline std::atomic<long long> prune_outcome[kPruneDepths][kPruneOutcomes] = {};
    inline std::atomic<long long> margin_searched[kPruneDepths][kMarginBuckets] = {};
    inline std::atomic<long long> margin_fail_high[kPruneDepths][kMarginBuckets] = {};
    inline std::atomic<long long> margin_subtree[kPruneDepths][kMarginBuckets] = {};
    // Declenchements du mecanisme appris (coupe, reduction ou ombre), par profondeur.
    inline std::atomic<long long> learned_fires[kPruneDepths] = {};

#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline bool prune_stats_enabled()
    {
        static const bool on = std::getenv("ALCYON_PRUNE_STATS") != nullptr;
        return on;
    }
#else
    constexpr bool prune_stats_enabled() { return false; }
#endif

    inline void record_prune(int depth, PruneOutcome o, int margin, long long subtree)
    {
        const int d = std::min(depth, kPruneDepths) - 1;
        prune_outcome[d][o].fetch_add(1, std::memory_order_relaxed);
        if (o != PO_FAIL_HIGH && o != PO_FAIL_LOW)
            return;
        int b = 0;
        while (b < kMarginBuckets - 1 && margin >= kMarginEdges[b])
            ++b;
        margin_searched[d][b].fetch_add(1, std::memory_order_relaxed);
        margin_subtree[d][b].fetch_add(subtree, std::memory_order_relaxed);
        if (o == PO_FAIL_HIGH)
            margin_fail_high[d][b].fetch_add(1, std::memory_order_relaxed);
    }

    // Dump (ALCYON_PRUNE_DUMP=<fichier>, ALCYON_PRUNE_DUMP_EVERY=N, voir
    // prune_dump_sample) : etape 1
    // de docs/learned-pruning.md. Un enregistrement par noeud tire au point
    // de decision (apres TT et RFP, avant NMP) : features scalaires a
    // l'entree, resultat et taille du sous-arbre a la sortie. Format binaire
    // brut, lu par tools/learned_pruning/fit.py qui en repete la
    // disposition -- tout changement ici doit y etre reporte.
    //
    // ponytail: un seul fichier, un seul tampon, non synchronise -- Threads=1
    // uniquement ; pour paralleliser, lancer plusieurs processus.
    struct PruneRecord
    {
        std::int64_t subtree;
        std::int32_t beta, static_eval, eval_prev2; // eval_prev2 = kEvalNone si inconnue
        std::int32_t tt_found, tt_score, tt_depth, tt_flag;
        std::int32_t depth, ply, outcome; // outcome : 0 fail-high, 1 fail-low, 2 NMP, 3 elagage appris
        std::int32_t cut_node, allow_null, halfmove, stm;
        std::int32_t prev_from_piece, prev_to_piece; // NO_PIECE si pas de capture
        std::int32_t us[5], them[5];                 // pions, cavaliers, fous, tours, dames
        float learned_z;                             // logit calcule par le moteur (verif. de parite)
        std::int32_t research;                       // noeud atteint par une re-recherche LMR (0 dans les dumps anciens)
        // v3 -- label "reduire est sur" (ALCYON_PRUNE_DUMP_LABEL=1) : score et
        // cout en noeuds de ce meme noeud cherche a depth-1 (indice 0) et
        // depth-2 (indice 1), meme fenetre, avant la recherche normale.
        std::int32_t red_score[2];
        std::int32_t red_nodes[2];
    };
    static_assert(sizeof(PruneRecord) == 136, "reporter la disposition dans fit.py (DTYPE_V3)");

    // Fichier parallele "<dump>.l0" : pour chaque enregistrement, l'entree
    // l0 de la pile de couches NNUE au point de decision (build NNUE).
    constexpr int kL0 = 1024;
    using L0 = std::array<std::uint8_t, kL0>;

    struct PruneDump
    {
        static constexpr int kBuf = 4096;
        std::FILE *f = nullptr;
        std::FILE *f_l0 = nullptr;
        L0 l0_buf[kBuf];
        // l0 capture au point de decision, en attendant la sortie du noeud.
        // ponytail: une case par ply -- suppose pas de noeud tire imbrique au
        // meme ply (vrai avec learned_prune_enabled=0, le reglage des dumps).
        L0 l0_at_ply[engine_constants::search::MaxDepth + 8];
        unsigned long long every = 1, counter = 0;
        int n = 0;
        PruneRecord buf[kBuf];

        void flush()
        {
            if (f && n)
                std::fwrite(buf, sizeof(PruneRecord), n, f);
            if (f_l0 && n)
                std::fwrite(l0_buf, sizeof(L0), n, f_l0);
            n = 0;
        }
        void push(const PruneRecord &r, int ply)
        {
            l0_buf[n] = l0_at_ply[ply];
            buf[n++] = r;
            if (n == kBuf)
                flush();
        }
        ~PruneDump()
        {
            flush();
            if (f)
                std::fclose(f);
            if (f_l0)
                std::fclose(f_l0);
        }
    };
    inline PruneDump prune_dump;

#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline bool prune_dump_enabled()
    {
        static const bool on = []
        {
            const char *path = std::getenv("ALCYON_PRUNE_DUMP");
            if (!path)
                return false;
            if (const char *e = std::getenv("ALCYON_PRUNE_DUMP_EVERY"))
                prune_dump.every = std::max(1ULL, std::strtoull(e, nullptr, 10));
            prune_dump.f = std::fopen(path, "wb");
            // Marqueur de format : fit.py lit "<dump>.v3" pour choisir DTYPE_V3
            // (les tailles 120 et 136 ne suffisent pas toujours a trancher).
            if (std::FILE *m = std::fopen((std::string(path) + ".v3").c_str(), "w"))
                std::fclose(m);
#ifdef NNUE_EVAL
            prune_dump.f_l0 = std::fopen((std::string(path) + ".l0").c_str(), "wb");
#endif
            return prune_dump.f != nullptr;
        }();
        return on;
    }
#else
    constexpr bool prune_dump_enabled() { return false; }
#endif

    // Tirage avec probabilite min(1, 2^(depth-1) / every) : les noeuds pres
    // des feuilles sont des millions, ceux de haute profondeur quelques
    // milliers -- un tirage uniforme remplirait le disque des premiers sans
    // donner assez des seconds. Les fits se font par tranche de profondeur,
    // donc ce biais entre profondeurs ne fausse rien a l'interieur d'une
    // tranche d'une seule profondeur.
    // Diagnostic (ALCYON_L0_CHECK=N) : un noeud de decision sur N, compare
    // l0 lu sur l'accumulateur incremental (paresseux) a l0 apres recalcul
    // complet de la position. La parite C++/Python ne peut pas voir un l0
    // perime : les deux cotes lisent le meme. Resultat dans "prunestats".
    inline std::atomic<long long> l0_checks{0}, l0_mismatches{0}, l0_max_diff{0};
#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline unsigned long long l0_check_every()
    {
        static const unsigned long long n = []
        {
            const char *e = std::getenv("ALCYON_L0_CHECK");
            return e ? std::max(1ULL, std::strtoull(e, nullptr, 10)) : 0ULL;
        }();
        return n;
    }
#else
    constexpr unsigned long long l0_check_every() { return 0; }
#endif

#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline bool prune_dump_label_enabled()
    {
        static const bool on = std::getenv("ALCYON_PRUNE_DUMP_LABEL") != nullptr;
        return on;
    }
#else
    constexpr bool prune_dump_label_enabled() { return false; }
#endif

    inline bool prune_dump_sample(int depth)
    {
        const unsigned long long keep = 1ULL << std::min(depth - 1, 40);
        return engine::random::splitmix64(prune_dump.counter++) % prune_dump.every < keep;
    }

#ifdef ALCYON_SEARCH_EXPERIMENTS
    inline bool tt_cutoffs_enabled()
    {
        static const bool on = std::getenv("ALCYON_TT_NO_CUTOFF") == nullptr;
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
    // Limite de noeuds de la recherche (0 = aucune) : bench nodesfile, pour
    // comparer deux configs a noeuds egaux position par position.
    long long node_limit = 0;
    int thread_id;

    Move best_root_move = 0;
    Move out_move = 0;

    int max_extended_depth;

    // Ply du noeud en cours de verification par l'elagage appris (reduction
    // verifiee, negamax.cpp) : la recherche reduite de ce noeud ne doit pas
    // redeclencher le mecanisme sur lui-meme. -1 = aucun.
    int learned_verify_ply = -1;

    // Le noeud du ply suivant est une re-recherche LMR (pose par
    // late_move_reduction_search, consomme en tete de negamax). L'elagage
    // appris s'en abstient : la TT y porte le resultat de la recherche
    // reduite qui vient de surprendre, et s'y fier annulerait la
    // verification de la LMR.
    bool research_node[engine_constants::search::MaxDepth + 8] = {};

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