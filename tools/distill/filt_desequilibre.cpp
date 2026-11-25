// Garde les positions desequilibrees (ecart de materiel 1-4 points) ou de finale
// (materiel hors pions < 20). Les positions consecutives d'une meme partie restent
// chainees par l'ecrivain binpack, donc la compression reste bonne.
// Usage : filt <entree.binpack> <sortie.binpack> [max_positions_gardees]
#include <cstdio>
#include <cstdlib>
#include "binpack.h"
using namespace chess;
int main(int c,char**v){
  binpack::CompressedTrainingDataEntryReader r(v[1],std::ios_base::in);
  binpack::CompressedTrainingDataEntryWriter w(v[2],std::ios_base::out);
  long long lim=c>3?std::atoll(v[3]):-1, n=0,k=0;
  const int val[6]={1,3,3,5,9,0};
  const PieceType pts[6]={PieceType::Pawn,PieceType::Knight,PieceType::Bishop,PieceType::Rook,PieceType::Queen,PieceType::King};
  while(r.hasNext()){
    auto e=r.next(); ++n;
    if(e.score==32002) continue;
    int m[2]={0,0}, npm=0;
    for(int i=0;i<6;++i) for(int col=0;col<2;++col){
      int cnt=e.pos.pieceCount(Piece(pts[i],col?Color::Black:Color::White));
      m[col]+=val[i]*cnt; if(i>0&&i<5) npm+=val[i]*cnt; }
    int d=std::abs(m[0]-m[1]);
    if((d>=1&&d<=4)||npm<20){ w.addTrainingDataEntry(e); if(++k==lim) break; }
  }
  std::fprintf(stderr,"lues %lld gardees %lld (%.1f %%)\n",n,k,100.0*k/n);
}
