#include "engine/eval/pos_eval.hpp"

namespace Eval
{
    namespace
    {
        constexpr int OutputScale = 400;
        constexpr int QA = 255;
        constexpr int QB = 64;

        inline int nnue_to_cp(int raw)
        {
            return (raw * OutputScale) / (QA * QB);
        }
    }

    int eval(const VBoard &board, int alpha, int beta)
    {
        const int raw = board.get_nnue_eval().evaluate_abs();

        return nnue_to_cp(raw);
    }
}