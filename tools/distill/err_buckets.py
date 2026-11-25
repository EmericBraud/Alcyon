import math, sys, chess, numpy as np
X=sys.argv[1] if len(sys.argv)>1 else None
NAMES=["v5","A","sf14"]+([X] if X else [])
from multiprocessing import Pool
W="/workspace/erran"; V={1:1,2:3,3:3,4:5,5:9,6:0}
def feats(l):
    t=l.rstrip("\n").split("\t"); fen,sc,res,ply=t[:4]; EV=[int(x) for x in t[4:]]; B=chess.Board(fen); us=B.turn
    npm={col:sum(V[p.piece_type] for p in B.piece_map().values() if p.color==col and p.piece_type not in (1,6)) for col in (0,1)}
    pw={col:len(B.pieces(1,col)) for col in (0,1)}
    tot=npm[0]+npm[1]
    phase="ouverture/milieu (>=40)" if tot>=40 else "milieu leger (20-39)" if tot>=20 else "finale (<20)"
    imb=(npm[us]+pw[us])-(npm[not us]+pw[not us])
    q={col:len(B.pieces(5,col)) for col in (0,1)}; r={col:len(B.pieces(4,col)) for col in (0,1)}
    minor={col:len(B.pieces(2,col))+len(B.pieces(3,col)) for col in (0,1)}
    # desequilibre de type de pieces (meme materiel ou non)
    if q[0]!=q[1]: typ="dames inegales"
    elif r[0]!=r[1]: typ="tours inegales (qualite...)"
    elif minor[0]!=minor[1]: typ="pieces legeres inegales"
    elif npm[0]==npm[1]: typ="pieces identiques"
    else: typ="autre"
    if tot==0: eg="finale de pions"
    elif tot<20 and q[0]+q[1]==0 and minor[0]+minor[1]==0: eg="finale de tours"
    elif tot<20 and q[0]+q[1]==0 and r[0]+r[1]==0: eg="finale de pieces legeres"
    elif tot<20 and q[0]+q[1]>0: eg="finale avec dames"
    elif tot<20: eg="finale mixte"
    else: eg="-"
    # pions passes
    def passed(col):
        n=adv=0
        for s in B.pieces(1,col):
            f,rk=chess.square_file(s),chess.square_rank(s)
            ahead=range(rk+1,8) if col else range(0,rk)
            if not any(B.piece_at(chess.square(ff,rr))==chess.Piece(1,not col) for ff in (f-1,f,f+1) if 0<=ff<8 for rr in ahead):
                n+=1; adv+= (rk>=5) if col else (rk<=2)
        return n,adv
    p0,p1=passed(0),passed(1)
    pp="pion passe avance (6e/7e)" if p0[1]+p1[1] else "pion passe" if p0[0]+p1[0] else "aucun passe"
    # tension : pieces attaquees par moins cher ou non defendues
    hang=0
    for s,p in B.piece_map().items():
        if p.piece_type==6: continue
        att=B.attackers(not p.color,s)
        if att and (not B.attackers(p.color,s) or min(V[B.piece_at(x).piece_type] or 99 for x in att)<V[p.piece_type]): hang+=1
    ten="0 piece en prise" if hang==0 else "1 piece en prise" if hang==1 else "2+ pieces en prise"
    # securite du roi : attaques adverses sur la zone du roi (camp au trait et adverse, max)
    def kz(col):
        k=B.king(col); z=chess.SquareSet(chess.BB_KING_ATTACKS[k])|chess.SquareSet.from_square(k)
        return sum(len(B.attackers(not col,s)) for s in z)
    ks=max(kz(0),kz(1)); kss="roi tres expose (>=6 att.)" if ks>=6 else "roi expose (3-5)" if ks>=3 else "roi tranquille (<3)"
    sc=int(sc); asc=abs(sc)
    ev="equilibre <50" if asc<50 else "leger 50-150" if asc<150 else "net 150-400" if asc<400 else "decisif >=400"
    pl=int(ply); plb="ply <20" if pl<20 else "ply 20-59" if pl<60 else "ply 60-119" if pl<120 else "ply >=120"
    return sc,EV,dict(phase=phase,typ=typ,finale=eg,passes=pp,tension=ten,roi=kss,eval=ev,ply=plb,
        materiel=("materiel egal" if imb==0 else "ecart 1-2" if abs(imb)<=2 else "ecart >=3"))
if __name__=="__main__":
    with Pool(62) as p: R=p.map(feats,open(f"{W}/evals{'_'+X if X else ''}.tsv"),chunksize=2000)
    sc=np.array([r[0] for r in R],float); E=np.array([r[1] for r in R],float); J=len(NAMES)
    sig=lambda x:1/(1+np.exp(-x))
    T=sig(sc/400)
    # echelle par reseau ajustee (l'eval moteur n'est pas forcement a l'echelle des scores Leela)
    ks=[]
    for j in range(J):
        best=min(np.arange(0.5,2.01,0.02),key=lambda k:np.mean((sig(k*E[:,j]/400)-T)**2)); ks.append(best)
    P=np.stack([sig(ks[j]*E[:,j]/400) for j in range(J)],1)
    err=(P-T[:,None])**2*1e4
    print("echelles ajustees",NAMES,[round(k,2) for k in ks])
    tot_gap=(err[:,1]-err[:,2]).sum()
    print(f"ensemble  n={len(R)}  erreur (x1e4) "+"  ".join(f"{n} {err[:,j].mean():.1f}" for j,n in enumerate(NAMES))+"\n")
    for key in ["phase","finale","materiel","typ","passes","tension","roi","eval","ply"]:
        print(f"== {key}"); print(f"  {'categorie':30s} {'part':>6s} {'v5':>6s} {'A':>6s} {'sf14':>6s} {'gain v5->A':>10s} {'reste A-sf14':>12s} {'part du reste':>13s}"+(f" {X:>6s} {X+' vs A':>9s}" if X else ""))
        cats={}
        for i,r in enumerate(R): cats.setdefault(r[2][key],[]).append(i)
        for c,ix in sorted(cats.items(),key=lambda kv:-len(kv[1])):
            e=err[ix]; m=e.mean(0); g=(e[:,1]-e[:,2]).sum()/tot_gap
            print(f"  {c:30s} {100*len(ix)/len(R):5.1f}% {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f} {100*(1-m[1]/m[0]):9.0f}% {100*(1-m[2]/m[1]):11.0f}% {100*g:12.0f}%"+(f" {m[3]:6.1f} {100*(m[3]/m[1]-1):+8.1f}%" if X else ""))
        print()
