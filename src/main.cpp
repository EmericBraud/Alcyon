#include "interface/uci.hpp"

#include <charconv>
#include <string>
#include <cstring>

static bool parse_int_arg(const char *s, int &out)
{
    const char *begin = s;
    const char *end = s + std::char_traits<char>::length(s);

    auto [ptr, ec] = std::from_chars(begin, end, out);
    return ec == std::errc() && ptr == end;
}

int main(int argc, char **argv)
{
    UCI u;

    if (argc >= 2)
    {
        std::string cmd = argv[1];

        if (cmd == "bench")
        {
            int bench_depth = 4;

            if (argc >= 3 && !parse_int_arg(argv[2], bench_depth))
                bench_depth = 4;

            u.run_bench_cli(bench_depth);
            return 0;
        }
    }

    u.loop();
    return 0;
}