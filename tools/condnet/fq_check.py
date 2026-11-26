"""Evals PyTorch en flottant et en quantification simulee (v7, WS2) sur une liste de FEN, et bornes
des poids des piles conditionnelles. Usage : fq_check.py <fens.txt> <modele_cond.pt> <sortie.tsv>"""
import sys, torch, statistics as st
import data_loader
from model.cond_model import cond_mask
fens = [l.strip() for l in open(sys.argv[1])]
out = {}
for name, p in [("v7", "/workspace/nets/v7.pt"), ("C", sys.argv[2])]:
    m = torch.load(p, weights_only=False, map_location="cpu").model.eval()
    bt = data_loader.get_sparse_batch_from_fens(m.input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
    us, them, wi, bi, o, s, pc = bt.contents.get_tensors("cpu")
    with torch.no_grad():
        for fq in (False, True):
            out[(name, fq)] = (m(us, them, wi, bi, pc, fq, fq).flatten() * m.quantization.nnue2score).round().int().tolist()
    if name == "C":
        c = cond_mask(wi, m.cond_name).tolist()
        cs = m.cond_stacks; q = m.quantization
        print("max |w| l1", cs.l1.linear.weight.abs().max().item(), "l1_extra", cs.l1_extra.linear.weight.abs().max().item(),
              "l2", cs.l2.linear.weight.abs().max().item(), "sortie", cs.output.linear.weight.abs().max().item(), "limites", q.max_hidden_weight)
        print("max |w| FT bloc", max(f.weight.abs().max().item() for f in m.cond_input.features), "FT v7", max(f.weight.abs().max().item() for f in m.input.features))
with open(sys.argv[3], "w") as f:
    for i, fen in enumerate(fens):
        f.write("\t".join([fen] + [str(out[k][i]) for k in [("v7", False), ("v7", True), ("C", False), ("C", True)]] + [str(int(c[i]))]) + "\n")
print("ecart flottant / quantif simulee, v7 :", round(st.mean(abs(x - y) for x, y in zip(out[("v7", False)], out[("v7", True)])), 2))
print("ecart flottant / quantif simulee, bloc sous condition :",
      round(st.mean(abs(x - y) for x, y, k in zip(out[("C", False)], out[("C", True)], c) if k), 2))
