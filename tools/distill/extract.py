import csv, sys, json
import chess, chess.pgn
rows=list(csv.DictReader(open(sys.argv[1])))
games=[]; f=open(sys.argv[2])
while (g:=chess.pgn.read_game(f)): games.append(g)
pos={}
for gi,g in enumerate(games):
    b=g.board()
    for n in g.mainline():
        pos[(gi,b.ply())]=(b.fen(), b.is_capture(n.move), b.gives_check(n.move), b.is_check()); b.push(n.move)
out=[]
for r in rows:
    es0,es1=float(r["es_before"]),float(r["es_after"])
    fen,cap,chk,inc=pos[(int(r["game"]),int(r["ply"]))]
    if 0.05<es0<0.95 and es0-es1>=0.10 and not cap and not chk and not inc:
        out.append(dict(fen=fen, move=r["move"], best=r["best"], loss=es0-es1, es0=es0, mat=int(r["mat"]), depth=int(r["depth"]), time=float(r["time"])))
json.dump(out, open(sys.argv[3],"w"))
print(len(out), "positions ; dont ouverture (mat>=50):", sum(o["mat"]>=50 for o in out))
