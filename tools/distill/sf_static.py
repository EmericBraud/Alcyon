"""Plancher : eval statique de SF vs sa propre recherche profondeur 9 sur les memes feuilles,
compare a l'eval statique d'Alcyon. En probabilite de gain (sigmoide v/340), feuilles calmes."""
import glob, gzip, math, os, random, statistics as st, subprocess, sys
from multiprocessing import Pool
import chess
wp=lambda v: 1/(1+math.exp(-max(-6000,min(6000,v))/340))
SF=os.path.expanduser("~/sfraw/src/stockfish")
def load(pat,n):
    files=glob.glob(os.path.expanduser(pat)); random.seed(1); random.shuffle(files); out=[]
    for f in files:
        for l in gzip.open(f,"rt"):
            fen,mv,sf,npv,stv,a,r=l.rstrip("\n").split("\t")
            b=chess.Board(fen); m=chess.Move.from_uci(mv)
            if b.is_capture(m) or m.promotion: continue
            out.append((fen,int(sf),int(stv)))
        if len(out)>=n: break
    return out[:n]
def sf_static(batch):
    inp="".join(f"position fen {f}\neval\n" for f,_,_ in batch)+"quit\n"
    o=subprocess.run([SF],input=inp,capture_output=True,text=True).stdout
    vals=[round(float(l.split()[2])*100) for l in o.splitlines() if l.startswith("Final evaluation")]
    res=[]
    for (f,sf,al),v in zip(batch,vals):
        stm=chess.Board(f).turn
        res.append((sf,al,v if stm==chess.WHITE else -v))
    return res
for name,pat in [("run2 reference",sys.argv[1]),("run1 desaccords",sys.argv[2])]:
    L=load(pat,int(sys.argv[3]))
    batches=[L[i:i+500] for i in range(0,len(L),500)]
    with Pool(32) as p: R=[x for b in p.map(sf_static,batches) for x in b]
    print(f"\n== {name} : {len(R)} feuilles calmes")
    print("  |P| SF recherche     n   | Alcyon statique |dP| | SF statique |dP| | part irreductible")
    buckets={}
    for sf,al,ss in R: buckets.setdefault(min(4,int(abs(wp(sf)-.5)*10)),[]).append((sf,al,ss))
    for k in sorted(buckets)+["all"]:
        S=R if k=="all" else buckets[k]
        ea=st.mean(abs(wp(al)-wp(sf)) for sf,al,ss in S); es=st.mean(abs(wp(ss)-wp(sf)) for sf,al,ss in S)
        lab="ensemble" if k=="all" else f"{50-10*k-10 if k<4 else 0}-{50-10*k}/{50+10*k}-{50+10*k+10 if k<4 else 100}%"
        print(f"  {lab:18s} {len(S):6d} |   {100*ea:5.1f}            |  {100*es:5.1f}          | {100*es/ea:4.0f} %")
