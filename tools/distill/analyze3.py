import re, sys, statistics as st
import chess, chess.pgn
CMT = re.compile(r"([+-]?)(M?)(\d+(?:\.\d+)?)/(\d+) ([\d.]+)s")
games=[]
with open(sys.argv[1]) as f:
    while (g:=chess.pgn.read_game(f)): games.append(g)
left_a, left_s, zero, zero_bad = [], [], 0, []
tot=0
for gi,g in enumerate(games):
    aw=g.headers["White"].startswith("alcyon")
    b=g.board(); ta=ts=0; na=ns=0; last_sf=None; pend=None
    for node in g.mainline():
        am=(b.turn==chess.WHITE)==aw
        m=CMT.search(node.comment or "")
        if m:
            sign,mate,v,d,t=m.groups(); t=float(t); cp=float(v)*100*(-1 if sign=="-" else 1)
            if am:
                ta+=t; na+=1; tot+=1
                if cp==0 and not mate:
                    zero+=1; pend=(b.ply(), b.san(node.move), int(d), t, b.fen())
                else: pend=None
            else:
                ts+=t; ns+=1
                sfpov=-cp
                if pend and sfpov <= -100:
                    zero_bad.append((gi+1,)+pend+(sfpov,))
                pend=None
        b.push(node.move)
    left_a.append(60+0.2*na-ta); left_s.append(60+0.2*ns-ts)
print(f"temps restant en fin de partie : Alcyon moyenne {st.mean(left_a):.1f}s (mediane {st.median(left_a):.1f}) | SF8 {st.mean(left_s):.1f}s (mediane {st.median(left_s):.1f})")
print(f"coups d'Alcyon avec eval exactement 0.00 : {zero}/{tot} ({100*zero/tot:.1f} %)")
print(f"  dont suivis d'une eval SF8 <= -100 (SF8 pense qu'Alcyon perd) : {len(zero_bad)}")
for z in zero_bad[:20]: print("  p%d ply %d %s d%d %.2fs SF8 %+.0f  %s"%(z[0],z[1],z[2],z[3],z[4],z[6],z[5]))
