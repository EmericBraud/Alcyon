#include <cstdio>
#include "binpack.h"
int main(int c,char**v){ binpack::CompressedTrainingDataEntryReader r(v[1],std::ios_base::in);
 while(r.hasNext()){auto e=r.next(); std::printf("%s\t%d\t%d\t%s\t%u\n",e.pos.fen().c_str(),e.score,e.result,chess::uci::moveToUci(e.pos,e.move).c_str(),e.ply);} }
