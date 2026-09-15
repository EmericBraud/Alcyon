// Inspecte un .binpack avec le lecteur officiel de nnue-pytorch
// (external/nnue-pytorch/data_loader/cpp/lib) plutot qu'un decodage maison.
//
// Repond a une question precise : le fichier contient-il un coup par position,
// et ce coup est-il legal dans la position stockee ? TrainingDataEntry::
// isValid() fait exactement cette verification (pos.isMoveLegal(move)).
//
// Build : voir tools/build_inspect_binpack.sh
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <vector>
#include <string>

#include "binpack.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fichier.binpack> [n_entrees]\n", argv[0]);
        return 1;
    }
    const std::string path = argv[1];
    const long long limit = (argc > 2) ? std::atoll(argv[2]) : 200000;

    binpack::CompressedTrainingDataEntryReader reader(path, std::ios_base::in);

    long long n = 0, legal = 0, null_move = 0, ply_under_10 = 0;
    long long score_abs_sum = 0, scored = 0, sentinel = 0, mates = 0, games = 0, max_ply = 0;
    std::vector<int> absv;
    std::map<int, long long> results;

    std::printf("=== 12 premieres entrees ===\n");
    while (reader.hasNext() && n < limit) {
        const binpack::TrainingDataEntry e = reader.next();

        if (n < 12) {
            std::printf("  ply=%4u  score=%6d  result=%+d  move=%-6s  %s\n", e.ply, e.score, e.result,
                        chess::uci::moveToUci(e.pos, e.move).c_str(), e.pos.fen().c_str());
        }

        if (e.isValid()) ++legal;
        if (e.move == chess::Move::null()) ++null_move;
        if (e.ply < 10) ++ply_under_10;
        if (e.ply == 0) ++games;
        if (e.ply > max_ply) max_ply = e.ply;
        // 32002 = VALUE_NONE cote Stockfish : la position n'a pas de score
        // utilisable (typiquement le premier coup d'une partie). Le COUP reste
        // valide, seul le label value manque.
        const int a = (e.score < 0 ? -e.score : e.score);
        if (e.score == 32002) {
            ++sentinel;            // VALUE_NONE : aucun label value utilisable
        } else {
            if (a > 30000) ++mates;  // score de mat : label legitime
            score_abs_sum += a; ++scored; absv.push_back(a);
        }
        ++results[e.result];
        ++n;
    }

    std::printf("\n=== sur %lld entrees ===\n", n);
    std::printf("  coups legaux dans leur position : %lld / %lld (%.4f %%)\n", legal, n, 100.0 * legal / n);
    std::printf("  coups nuls                      : %lld\n", null_move);
    std::printf("  score == VALUE_NONE (32002)     : %lld (%.3f %%) -- pas de label value\n", sentinel, 100.0 * sentinel / n);
    std::printf("  scores de mat (|s|>30000)       : %lld (%.3f %%) -- labels legitimes\n", mates, 100.0 * mates / n);
    std::sort(absv.begin(), absv.end());
    auto q=[&](double f){ return absv.empty()?0:absv[(std::size_t)(f*(absv.size()-1))]; };
    std::printf("  |score| median / p75 / p90 / p99 : %d / %d / %d / %d cp\n", q(0.5), q(0.75), q(0.90), q(0.99));
    std::printf("  |score| moyen                   : %.1f cp\n", double(score_abs_sum) / (scored ? scored : 1));
    std::printf("  parties (ply==0)                : %lld  | ply max vu : %lld\n", games, max_ply);
    std::printf("  positions a ply < 10            : %lld (%.2f %%)\n", ply_under_10, 100.0 * ply_under_10 / n);
    std::printf("  resultats                       : ");
    for (const auto &[r, c] : results) std::printf("%+d:%lld  ", r, c);
    std::printf("\n");
    return 0;
}
