#include "engine/eval/pos_eval.hpp"
namespace Eval
{
    int eval(const VBoard &board, int alpha, int beta)
    {
        return board.get_nnue_eval().evaluate_abs();
    }

}
