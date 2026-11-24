"""Ecart eval statique Alcyon / SF sur les feuilles, en probabilite de gain
(sigmoide de l'entraineur : p = 1/(1+exp(-v/340)), unites internes), feuilles calmes
(meilleur coup SF pas une capture/promotion)."""
import glob, gzip, math, os, random, statistics as st, sys
import chess
VAL={chess.PAWN:1,chess.KNIGHT:3,chess.BISHOP:3,chess.ROOK:5,chess.QUEEN:9}
wp=lambda v: 1/(1+math.exp(-max(-6000,min(6000,v))/340))
def load(pat, n):
    files=glob.glob(os.path.expanduser(pat)); random.seed(0); random.shuffle(files)
    out=[]
    for f in files:
        for l in gzip.open(f,"rt"):
            fen,mv,sf,npv,stv,a_sc,r=l.rstrip("\n").split("\t")
            out.append((fen,mv,int(sf),int(stv)))
        if len(out)>=n: break
    return out[:n]
def quiet(fen,mv):
    b=chess.Board(fen); m=chess.Move.from_uci(mv)
    return not b.is_capture(m) and m.promotion is None
def ph(fen):
    b=chess.Board(fen); m=sum(VAL.get(p.piece_type,0) for p in b.piece_map().values())
    return "1 >=50" if m>=50 else "2 30-49" if m>=30 else "3 12-29" if m>=12 else "4 <12"
for name,pat in [("run2 (tous les coups, reference)",sys.argv[1]),("run1 (desaccords seulement)",sys.argv[2])]:
    R=load(pat,300000)
    Q=[r for r in R if quiet(r[0],r[1])]
    print(f"\n== {name} : {len(R)} feuilles, calmes {len(Q)} ({100*len(Q)/len(R):.0f}%)")
    for lab,S in [("toutes",R),("calmes",Q)]:
        e=[abs(wp(s)-wp(x)) for _,_,x,s in S]
        b=[wp(s)-wp(x) for _,_,x,s in S]
        print(f"  {lab:7s} ecart |dP| moyen {100*st.mean(e):.1f} pts, median {100*st.median(e):.1f} ; >10 pts {100*sum(v>.1 for v in e)/len(e):.0f}% ; >25 pts {100*sum(v>.25 for v in e)/len(e):.0f}% ; biais {100*st.mean(b):+.1f} pts")
    by={}
    for fen,mv,x,s in Q: by.setdefault(ph(fen),[]).append(wp(s)-wp(x))
    print("  calmes par phase : n | |dP| moyen | biais")
    for k in sorted(by): v=by[k]; print(f"    {k:8s} {len(v):6d} | {100*st.mean(map(abs,v)):.1f} | {100*st.mean(v):+.1f}")
    by={}
    for fen,mv,x,s in Q: by.setdefault(min(4,int(abs(wp(x)-.5)*10)),[]).append(wp(s)-wp(x))
    print("  calmes par proba SF (distance a 50 %) : n | |dP| moyen | biais")
    for k in sorted(by): v=by[k]; print(f"    {50-10*k-10 if k<4 else 0:2d}-{50-10*k:2d}% / {50+10*k:2d}-{50+10*k+10 if k<4 else 100}%  {len(v):6d} | {100*st.mean(map(abs,v)):.1f} | {100*st.mean(v):+.1f}")
