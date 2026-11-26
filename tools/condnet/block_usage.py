"""Utilisation du bloc conditionnel entraine (variante B ou W) sur les positions calmes sous
condition de la validation : neurones morts, dimension utilisee (ACP), apport de chaque neurone
(extinction un par un), et part du gain qui vient du bloc ou des piles propres.
Usage (depuis la racine de np) : block_usage.py <init.pt> <ckpt> <val.tsv> [n_max]"""
import sys, torch, chess
import data_loader
from model.cond_model import cond_mask

init, ck, tsv = sys.argv[1], sys.argv[2], sys.argv[3]
n_max = int(sys.argv[4]) if len(sys.argv) > 4 else 200000
nnue = torch.load(init, weights_only=False, map_location="cpu")
nnue.load_state_dict(torch.load(ck, weights_only=False, map_location="cpu")["state_dict"])
m = nnue.model.cuda().eval()
v7 = torch.load("/workspace/nets/v7.pt", weights_only=False, map_location="cpu").model.cuda().eval()
S = m.quantization.nnue2score

rows = []
for l in open(tsv):
    fen, sc, res, mv, ply = l.rstrip("\n").split("\t"); sc = int(sc)
    if sc == 32002 or abs(sc) >= 10000: continue
    b = chess.Board(fen); mo = chess.Move.from_uci(mv)
    if b.is_check() or b.is_capture(mo) or mo.promotion: continue
    rows.append((fen, sc))
print("positions calmes", len(rows), flush=True)

sig = lambda x: torch.sigmoid(x / 400)
LB, TGT, OUT = [], [], {"v7": [], "B": [], "B sans bloc": []}
ctx = []
with torch.no_grad():
    for i in range(0, len(rows), 8192):
        fens = [r[0] for r in rows[i:i + 8192]]
        bt = data_loader.get_sparse_batch_from_fens(m.input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
        us, them, wi, bi, o, s, pc = bt.contents.get_tensors("cuda")
        keep = cond_mask(wi, m.cond_name)
        if keep.sum() == 0:
            data_loader.destroy_sparse_batch(bt); continue
        sc = torch.tensor([r[1] for r in rows[i:i + 8192]], device="cuda", dtype=torch.float32)
        psqt, ls = m.calculate_buckets(pc)
        l0, wp, bp = m.forward_ft(us, them, wi, bi, psqt, False, False)
        lb, _, _ = m.cond_input(us, them, wi, bi, torch.zeros_like(psqt), False, False)
        ps = (wp - bp) * (us - 0.5)
        out_b = m.cond_stacks(l0, lb, ls, False, False) + ps
        out_0 = m.cond_stacks(l0, torch.zeros_like(lb), ls, False, False) + ps
        out_v = v7(us, them, wi, bi, pc, False, False)
        k = keep
        LB.append(lb[k]); TGT.append(sc[k])
        OUT["B"].append(out_b[k].flatten() * S); OUT["B sans bloc"].append(out_0[k].flatten() * S); OUT["v7"].append(out_v[k].flatten() * S)
        ctx.append((l0[k], ls[k], ps[k]))
        data_loader.destroy_sparse_batch(bt)
        if sum(x.shape[0] for x in LB) >= n_max: break
LB = torch.cat(LB); T = sig(torch.cat(TGT))
err = lambda e: ((sig(e) - T) ** 2).mean().item() * 1e4
print("positions sous condition", LB.shape[0])
for n, v in OUT.items():
    print(f"  erreur {n:12s} {err(torch.cat(v)):.2f}")

act = (LB > 0).float().mean(0)
print("neurones morts (actifs < 0,1 % du temps) :", int((act < 0.001).sum()), "sur", LB.shape[1])
print("activite par neurone : mediane", round(act.median().item(), 3), "min", round(act.min().item(), 4), "max", round(act.max().item(), 3))
X = LB - LB.mean(0)
ev = torch.linalg.eigvalsh((X.T @ X) / X.shape[0]).flip(0).clamp(min=0)
cum = ev.cumsum(0) / ev.sum()
for q in (0.5, 0.9, 0.99):
    print(f"composantes pour {int(q * 100)} % de la variance :", int((cum < q).sum()) + 1)

base = err(torch.cat(OUT["B"]))
gains = []
with torch.no_grad():
    for j in range(LB.shape[1]):
        outs, off = [], 0
        for (l0, ls, ps) in ctx:
            n = l0.shape[0]; lb = LB[off:off + n].clone(); lb[:, j] = 0; off += n
            outs.append((m.cond_stacks(l0, lb, ls, False, False) + ps).flatten() * S)
        gains.append(err(torch.cat(outs)) - base)
g = torch.tensor(gains)
tot = err(torch.cat(OUT["B sans bloc"])) - base
print(f"apport total du bloc (erreur sans bloc - avec) : {tot:.3f}")
gs = g.sort(descending=True).values
print("apport des neurones (erreur en plus si eteint) : top 10", [round(x, 3) for x in gs[:10].tolist()])
print("somme des 10 premiers / somme totale :", round((gs[:10].clamp(min=0).sum() / gs.clamp(min=0).sum()).item(), 2))
print("neurones dont l'extinction coute plus de 1 % de l'apport total :", int((g > 0.01 * tot).sum()))
