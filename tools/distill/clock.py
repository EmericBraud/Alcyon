import re, sys, statistics as st
from collections import defaultdict
import chess.pgn
CMT=re.compile(r"/(\d+) ([\d.]+)s")
ca,cs,ta,tsf=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
with open(sys.argv[1]) as f:
    while (g:=chess.pgn.read_game(f)):
        aw=g.headers["White"].startswith("alcyon"); a=s=60.0; ply=0
        for node in g.mainline():
            m=CMT.search(node.comment or ""); am=(ply%2==0)==aw
            if m:
                t=float(m.group(2)); mv=ply//2+1
                if am: a+=0.2-t; ta[mv//10].append(t)
                else: s+=0.2-t; tsf[mv//10].append(t)
                if ply%2==1 and mv%10==0: ca[mv].append(a); cs[mv].append(s)
            ply+=1
print("coup | pendule Alcyon / SF8 (s, mediane) | temps/coup moyen sur la dizaine Alcyon / SF8")
for mv in range(10,130,10):
    if len(ca[mv])>=20: print(f"{mv:4d} | {st.median(ca[mv]):5.1f} / {st.median(cs[mv]):5.1f}  (n={len(ca[mv])}) | {st.mean(ta[mv//10-1]):.2f} / {st.mean(tsf[mv//10-1]):.2f}")
