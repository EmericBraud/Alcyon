"""v7.pt -> cond.pt : ajoute le bloc conditionnel (a lancer depuis la racine de np)."""
import sys, torch
from model.cond_model import make_cond
src, dst = sys.argv[1], sys.argv[2]
block = int(sys.argv[3]) if len(sys.argv) > 3 else 128
cond = sys.argv[4] if len(sys.argv) > 4 else "typ"
nnue = torch.load(src, weights_only=False, map_location="cpu")
make_cond(nnue.model, "Full_Threats+HalfKAv2_hm^", block, cond)
torch.save(nnue, dst)
print("ecrit", dst, "bloc", block, "condition", cond)
