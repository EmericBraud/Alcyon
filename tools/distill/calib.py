"""Echelle : eval statique Alcyon (unites du reseau v5) vs SF brut (internes) et SF cp, depth 9."""
import random, subprocess, statistics as st, chess, chess.engine, os
random.seed(1)
import chess.pgn
allf=[]
f=open("v62_sf8.pgn")
while (g:=chess.pgn.read_game(f)):
    b=g.board()
    for n in g.mainline():
        if not b.is_check(): allf.append(b.fen())
        b.push(n.move)
fens=random.sample(allf,600)
env=dict(os.environ, ALCYON_DATA_DIR=os.path.expanduser("~/v62/build/data"))
inp="".join(f"position fen {f}\neval\n" for f in fens)+"quit\n"
out=subprocess.run([os.path.expanduser("~/v62/build/alcyon")],input=inp,capture_output=True,text=True,env=env).stdout
al=[int(x.split()[1]) for x in out.splitlines() if x.startswith("score:")]
raw=chess.engine.SimpleEngine.popen_uci(os.path.expanduser("~/sfraw/src/stockfish"))
cp=chess.engine.SimpleEngine.popen_uci(os.path.expanduser("~/sfdev/src/stockfish"))
R=[];C=[]
for f in fens:
    b=chess.Board(f)
    R.append(raw.analyse(b,chess.engine.Limit(depth=9))["score"].relative.score(mate_score=32000))
    C.append(cp.analyse(b,chess.engine.Limit(depth=9))["score"].relative.score(mate_score=32000))
raw.quit(); cp.quit()
def slope(x,y):
    p=[(a,b) for a,b,c in zip(x,y,C) if abs(c)<400]
    mx=st.mean(a for a,_ in p); my=st.mean(b for _,b in p)
    return sum((a-mx)*(b-my) for a,b in p)/sum((a-mx)**2 for a,_ in p), len(p)
print(len(fens), len(al))
print("SF brut ~= %.2f x SF cp (n=%d)"%slope(C,R))
print("SF brut ~= %.2f x Alcyon statique (n=%d)"%slope(al,R))
print("Alcyon ~= %.2f x SF brut"%slope(R,al)[0])
print("moyennes |.| : alcyon %.0f brut %.0f cp %.0f"%(st.mean(map(abs,al)),st.mean(map(abs,R)),st.mean(map(abs,C))))

import math
p=[(a,r) for a,r,c in zip(al,R,C) if abs(c)<400]
xa=[a for a,_ in p]; yr=[r for _,r in p]
ma,mr=st.mean(xa),st.mean(yr)
cor=sum((a-ma)*(r-mr) for a,r in p)/math.sqrt(sum((a-ma)**2 for a in xa)*sum((r-mr)**2 for r in yr))
print("correlation Alcyon/SF brut (|cp|<400): %.2f ; ratio des ecarts-types brut/alcyon %.2f"%(cor, st.pstdev(yr)/st.pstdev(xa)))
