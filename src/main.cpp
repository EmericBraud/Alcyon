#include "interface/uci.hpp"

int main()
{
    /*VBoard b;
    b.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1");
    EngineManager e{b};
    e.start_search(500);
    e.wait();*/
    UCI u;
    u.loop();
    return 0;
}