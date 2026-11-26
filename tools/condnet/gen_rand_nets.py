"""Reseaux a poids aleatoires (init par defaut de nnue-pytorch) de formes variees, pour mesurer la
vitesse d'Alcyon selon la largeur de l'accumulateur (L1) et de la tete (L2, L3). La vitesse ne depend
que des formes. Usage (depuis la racine de nnue-pytorch) : gen_rand_nets.py <dossier>"""
import sys, os, torch
import model as M
from model.config import ModelConfig
from model.model import NNUEModel
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
torch.manual_seed(0)
for L1, L2, L3 in [(1024, 32, 32), (1152, 128, 64), (1536, 128, 64), (1536, 32, 32), (1024, 128, 64), (1152, 32, 32)]:
    m = NNUEModel("Full_Threats+HalfKAv2_hm^", ModelConfig(L1=L1, L2=L2, L3=L3))
    w = M.NNUEWriter(m, "aleatoire, mesure de vitesse", ft_compression="none", verbose=False)
    p = f"{out}/rand_{L1}_{L2}_{L3}.nnue"; open(p, "wb").write(w.buf); print(p, len(w.buf))
