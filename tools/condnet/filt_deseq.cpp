// Garde les positions avec un desequilibre de materiel : pieces differentes (dames, tours ou
// pieces legeres en nombre different) ou nombre de pions different. Condition "deseq" du bloc.
// Saute les positions marquees (score 32002). Usage : filt_deseq <entree> <sortie> [max_gardees]
#include <cstdio>
#include <cstdlib>
#include "binpack.h"
using namespace chess;
int main(int c, char **v) {
  binpack::CompressedTrainingDataEntryReader r(v[1], std::ios_base::in);
  binpack::CompressedTrainingDataEntryWriter w(v[2], std::ios_base::out);
  long long lim = c > 3 ? std::atoll(v[3]) : -1, n = 0, k = 0;
  auto cnt = [](const Position &p, PieceType t, Color col) { return int(p.pieceCount(Piece(t, col))); };
  while (r.hasNext()) {
    auto e = r.next(); ++n;
    if (e.score == 32002) continue;
    const auto &p = e.pos;
    bool q = cnt(p, PieceType::Queen, Color::White) != cnt(p, PieceType::Queen, Color::Black);
    bool rk = cnt(p, PieceType::Rook, Color::White) != cnt(p, PieceType::Rook, Color::Black);
    bool mi = cnt(p, PieceType::Knight, Color::White) + cnt(p, PieceType::Bishop, Color::White) !=
              cnt(p, PieceType::Knight, Color::Black) + cnt(p, PieceType::Bishop, Color::Black);
    bool pw = cnt(p, PieceType::Pawn, Color::White) != cnt(p, PieceType::Pawn, Color::Black);
    if (q || rk || mi || pw) { w.addTrainingDataEntry(e); if (++k == lim) break; }
  }
  std::fprintf(stderr, "%s : lues %lld gardees %lld (%.1f %%)\n", v[1], n, k, 100.0 * k / n);
}
