import random, subprocess, chess
from multiprocessing import Pool
W="/workspace/erran"
NETS=[("v5","/workspace/run_v5.sh"),("A","/workspace/nets_anneal/A/run.sh"),("sf14","/workspace/run_sf14.sh")]
import sys
if len(sys.argv)>1: NETS.append((sys.argv[1],f"/workspace/nets_{sys.argv[1]}/run.sh"))
def filt(l):
    fen,sc,res,mv,ply=l.rstrip("\n").split("\t"); sc=int(sc)
    if sc==32002 or abs(sc)>=10000: return None
    b=chess.Board(fen); m=chess.Move.from_uci(mv)
    if b.is_check() or b.is_capture(m) or m.promotion: return None
    return (fen,sc,int(res),int(ply))
def run(args):
    cmd,batch=args
    inp="".join(f"position fen {r[0]}\neval\n" for r in batch)+"quit\n"
    o=subprocess.run([cmd],input=inp,capture_output=True,text=True).stdout
    v=[int(x.split()[1]) for x in o.splitlines() if x.startswith("score:")]
    assert len(v)==len(batch),(cmd,len(v)); return v
if __name__=="__main__":
    with Pool(62) as p:
        rows=[r for r in p.map(filt,open(f"{W}/val.tsv"),chunksize=5000) if r]
        print("positions",len(rows),flush=True)
        bs=[rows[i:i+2000] for i in range(0,len(rows),2000)]
        out={n:[x for b in p.map(run,[(c,b) for b in bs]) for x in b] for n,c in NETS}
    with open(f"{W}/evals{'_'+sys.argv[1] if len(sys.argv)>1 else ''}.tsv","w") as f:
        for i,r in enumerate(rows): f.write("\t".join(map(str,r))+"\t"+"\t".join(str(out[n][i]) for n,_ in NETS)+"\n")
    print("fini",flush=True)
