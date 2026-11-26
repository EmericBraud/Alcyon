"""Verifie : (1) comptage de pieces = piece_count du chargeur ; (2) part de positions ou la
condition est vraie ; (3) cond.pt donne les memes evals que v7.pt au depart."""
import sys, torch
import data_loader
from model.cond_model import piece_counts, cond_mask
v7 = torch.load(sys.argv[1], weights_only=False, map_location="cpu").model.cuda().eval()
cd = torch.load(sys.argv[2], weights_only=False, map_location="cpu").model.cuda().eval()
fens = [l.split("\t")[0] for l, _ in zip(open(sys.argv[3]), range(20000))]
bt = data_loader.get_sparse_batch_from_fens(v7.input_feature_name, fens, [0]*len(fens), [1]*len(fens), [0]*len(fens))
us, them, wi, bi, o, s, pc = bt.contents.get_tensors("cuda")
c = piece_counts(wi)
print("comptage = piece_count :", bool((c.sum(dim=(1, 2)) == pc.flatten()).all()), "| exemple", c[0].tolist(), int(pc.flatten()[0]))
print("part ou la condition est vraie :", round(cond_mask(wi, cd.cond_name).float().mean().item() * 100, 1), "%")
with torch.no_grad():
    a = v7(us, them, wi, bi, pc, False, False); b = cd(us, them, wi, bi, pc, False, False)
print("ecart max v7 / cond au depart :", (a - b).abs().max().item() * v7.quantization.nnue2score, "cp")
