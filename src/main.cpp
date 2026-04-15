#include "interface/uci.hpp"

int main(int argc, char **argv)
{
    /*VBoard b;
    b.load_fen("r1q1r1k1/3b1p1p/3p4/2p3p1/1p1Pn3/1P1PPQ2/P2PK1PP/R2RBB2 w - - 0 1");
    EngineManager e{b};
    e.start_search(500);
    e.wait();*/
    UCI u;

    if (argc >= 2)
    {
        std::string cmd = argv[1];
        if (cmd == "bench")
        {
            int bench_depth = 4;
            if (argc >= 3)
            {
                try
                {
                    bench_depth = std::stoi(argv[2]);
                }
                catch (...)
                {
                }
            }

            u.run_bench_cli(bench_depth);
            return 0;
        }
    }

    u.loop();
    return 0;
}