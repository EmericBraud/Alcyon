// FEN (une par ligne, sorties d'Alcyon avec ALCYON_DUMP_DIR) -> binpack de positions isolees, au
// format des donnees "arbres de recherche" de SF (xushawn/sf-searchtree-bt4) : score 0 (a remplacer
// par la reevaluation BT4), ply 9999 (marque une position d'arbre), resultat 0, et un coup legal
// non capturant arbitraire (le format en demande un ; il ne porte aucune information).
// Doublons supprimes (empreinte de la FEN). Positions sans coup calme legal ignorees.
// Usage : fen2binpack <sortie.binpack> < positions.fen
#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_set>
#include "binpack.h"
int main(int c, char **v) {
  binpack::CompressedTrainingDataEntryWriter w(v[1], std::ios_base::out);
  std::unordered_set<std::size_t> seen;
  std::string line;
  long long n = 0, k = 0, dup = 0, nomove = 0;
  while (std::getline(std::cin, line)) {
    ++n;
    if (!seen.insert(std::hash<std::string>{}(line)).second) { ++dup; continue; }
    binpack::TrainingDataEntry e;
    e.pos = chess::Position::fromFen(line);
    bool found = false;
    chess::movegen::forEachLegalMove(e.pos, [&](chess::Move m) {
      if (!found && e.pos.pieceAt(m.to) == chess::Piece::none() && m.type == chess::MoveType::Normal) { e.move = m; found = true; }
    });
    if (!found) { ++nomove; continue; }
    e.score = 0; e.ply = 9999; e.result = 0;
    w.addTrainingDataEntry(e); ++k;
  }
  std::fprintf(stderr, "lues %lld doublons %lld sans coup calme %lld ecrites %lld\n", n, dup, nomove, k);
}
