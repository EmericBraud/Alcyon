#include <bit>

#include "engine/eval/pos_eval.hpp"
#include "engine/eval/eval_cache.hpp"

#ifdef NNUE_EVAL

namespace Eval
{
    // Partagé entre tous les threads de recherche (voir la note lockless
    // dans eval_cache.hpp). Statique au TU : Eval::eval est l'unique point
    // d'entrée de l'éval complète, inutile de le faire transiter par les
    // workers.
    static EvalCache eval_cache;

    int eval(const VBoard &board, int alpha, int beta)
    {
        // L'éval statique ne dépend que de la position : la valeur cachée
        // est bit-identique à celle qu'evaluate_abs() recalculerait, un hit
        // saute tout le forward pass NNUE (materialize + couches denses)
        // sans changer quoi que ce soit au search.
        const std::uint64_t key = board.get_hash();
        int cached;
        if (eval_cache.probe(key, cached))
            return cached;

        // Uses VBoard's incrementally-maintained accumulator (see
        // virtual_board.hpp's play/unplay hooks) -- no per-call recompute.
        // evaluate_abs() returns centipawns relative to the side to move (the
        // network is always evaluated as if "us" = whoever is on move). This
        // function's contract (matching the HCE eval and eval_relative<Us> in
        // pos_eval.hpp) is to return a WHITE-relative absolute score, so flip
        // the sign back for black to move.
        const int score = board.get_nnue_eval().evaluate_abs(board);
        const int abs_score = board.get_side_to_move() == WHITE ? score : -score;
        eval_cache.store(key, abs_score);
        return abs_score;
    }
}

#endif