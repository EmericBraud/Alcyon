import csv, sys, statistics as st
from collections import defaultdict, Counter
import chess, chess.pgn
rows=list(csv.DictReader(open(sys.argv[1])))
games=[]; f=open(sys.argv[2])
while (g:=chess.pgn.read_game(f)): games.append(g)
pos={}
for gi,g in enumerate(games):
    b=g.board()
    for n in g.mainline():
        pos[(gi,b.ply())]=(b.copy(stack=False), n.move); b.push(n.move)
def ph(m): return "1 ouverture>=50" if m>=50 else "2 milieu 30-49" if m>=30 else "3 finale 12-29" if m>=12 else "4 finale <12"
for r in rows:
    for k in ["es_before","es_after","time"]: r[k]=float(r[k])
    for k in ["game","ply","depth","mat","cp_before","cp_after"]: r[k]=int(r[k])
    r["loss"]=max(0.0,r["es_before"]-r["es_after"])
    r["ph"]=ph(r["mat"])
    b,mv=pos[(r["game"],r["ply"])]
    r["cap"]=b.is_capture(mv); r["chk"]=b.gives_check(mv); r["piece"]=chess.piece_name(b.piece_type_at(mv.from_square))
    r["incheck"]=b.is_check(); r["nlegal"]=b.legal_moves.count()
# only "open" positions where outcome uncertain
open_=[r for r in rows if 0.05<r["es_before"]<0.95]
print(f"coups d'Alcyon : {len(rows)} ; en position non decidee (0.05<ES<0.95) : {len(open_)}")
print(f"perte moyenne de score attendu par coup (non decide) : {100*st.mean(r['loss'] for r in open_):.2f} %   accord avec SF (meme coup) : {100*sum(r['move']==r['best'] for r in rows)/len(rows):.1f} %")
def cls(l): return "gaffe >=20%" if l>=.2 else "erreur 10-20%" if l>=.1 else "imprecision 5-10%" if l>=.05 else "ok"
c=Counter(cls(r["loss"]) for r in open_); print("classes :", dict(c))
print("\npar phase : n | perte moy % | erreurs>=10% | gaffes>=20% | prof moy | temps moy | accord SF %")
for p in sorted(set(r["ph"] for r in open_)):
    R=[r for r in open_ if r["ph"]==p]
    print(f" {p:16s} {len(R):5d} | {100*st.mean(r['loss'] for r in R):.2f} | {sum(r['loss']>=.1 for r in R):4d} | {sum(r['loss']>=.2 for r in R):3d} | {st.mean(r['depth'] for r in R):.1f} | {st.mean(r['time'] for r in R):.2f}s | {100*sum(r['move']==r['best'] for r in R)/len(R):.0f}")
tot=sum(r["loss"] for r in open_)
print("\npart de la perte totale par phase :", {p: f"{100*sum(r['loss'] for r in open_ if r['ph']==p)/tot:.0f}%" for p in sorted(set(r['ph'] for r in open_))})
print("\npar temps passe sur le coup :")
for lo,hi in [(0,.25),(.25,.5),(.5,1),(1,2),(2,99)]:
    R=[r for r in open_ if lo<=r["time"]<hi]
    if R: print(f"  {lo}-{hi}s n={len(R):5d} perte {100*st.mean(r['loss'] for r in R):.2f}%  erreurs>=10% {100*sum(r['loss']>=.1 for r in R)/len(R):.1f}%")
print("\nerreurs >=10% par type de coup (taux) :")
for name,f in [("capture",lambda r:r["cap"]),("echec donne",lambda r:r["chk"]),("en echec",lambda r:r["incheck"]),("calme",lambda r:not r["cap"] and not r["chk"] and not r["incheck"])]:
    R=[r for r in open_ if f(r)]
    print(f"  {name:12s} n={len(R):5d} perte {100*st.mean(r['loss'] for r in R):.2f}% erreurs {100*sum(r['loss']>=.1 for r in R)/len(R):.1f}%")
print("par piece jouee :")
for pc in ["pawn","knight","bishop","rook","queen","king"]:
    R=[r for r in open_ if r["piece"]==pc]
    print(f"  {pc:7s} n={len(R):5d} perte {100*st.mean(r['loss'] for r in R):.2f}% erreurs {100*sum(r['loss']>=.1 for r in R)/len(R):.1f}%")
print("\nselon equilibre de la position (ES avant) :")
for lo,hi in [(.05,.25),(.25,.4),(.4,.6),(.6,.75),(.75,.95)]:
    R=[r for r in open_ if lo<=r["es_before"]<hi]
    print(f"  ES {lo}-{hi} n={len(R):5d} perte {100*st.mean(r['loss'] for r in R):.2f}% erreurs {100*sum(r['loss']>=.1 for r in R)/len(R):.1f}%")
# lost games: decisive move
res={}
for gi,g in enumerate(games):
    aw=g.headers["White"].startswith("alcyon"); s={"1-0":1,"0-1":0,"1/2-1/2":.5}[g.headers["Result"]]; res[gi]=s if aw else 1-s
print("\ndefaites/nulles : coup qui a coute le plus (score attendu) ")
for outcome,name in [(0,"defaites"),(.5,"nulles")]:
    G=[gi for gi in res if res[gi]==outcome]
    worst=[max((r for r in rows if r["game"]==gi), key=lambda r:r["loss"]) for gi in G]
    print(f"  {name} ({len(G)}) : perte max moyenne {100*st.mean(w['loss'] for w in worst):.0f}% ; phase :", dict(sorted(Counter(w['ph'] for w in worst).items())), "; ES de depart moyen", f"{st.mean(rows_[0]['es_before'] for rows_ in [[r for r in rows if r['game']==gi] for gi in G] if rows_):.2f}")
print("\nscore attendu au 1er coup d'Alcyon (ouverture UHO) selon resultat :")
for outcome in [0,.5,1]:
    G=[gi for gi in res if res[gi]==outcome]
    v=[min((r for r in rows if r["game"]==gi), key=lambda r:r["ply"])["es_before"] for gi in G]
    print(f"  resultat {outcome}: n={len(G)} ES depart moyen {st.mean(v):.2f}")
