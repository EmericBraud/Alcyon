"""Export d'un modele a bloc conditionnel (variante W : bloc de B, piles propres 128-64) vers deux
fichiers .nnue au format nnue-pytorch, lisibles par Alcyon :

  <sortie>.nnue      : accumulateur COMBINE de L1 = 1024 + B colonnes (v7 + bloc) et les 8 piles de
                       v7, dont les poids sur les colonnes du bloc sont nuls (= v7 exactement) ;
  <sortie>.cond.nnue : le meme accumulateur et les 8 piles propres aux positions sous condition
                       (L2 = 128, L3 = 64). Alcyon n'en lit que les piles.

Ordre des colonnes de l'accumulateur combine : le produit par paires associe i et i + L1/2, donc
  [v7 0..511, bloc 0..B/2-1, v7 512..1023, bloc B/2..B-1]
et l0 (sortie du produit par paires) = [nous : v7 512, bloc B/2 | eux : v7 512, bloc B/2].

Usage (depuis la racine de nnue-pytorch, avec model/cond_model.py) :
  export_cond.py <modele.pt> <sortie>  [--check <val.tsv>]
"""
import sys, torch
import model as M
from model.config import ModelConfig
from model.model import NNUEModel

src, out = sys.argv[1], sys.argv[2]
F = "Full_Threats+HalfKAv2_hm^"
cm = torch.load(src, weights_only=False, map_location="cpu").model.eval()
B = cm.cond_block
Lm = cm.input.l1_size            # 1024
N = Lm + B
H, h = Lm // 2, B // 2
cols_main = list(range(0, H)) + list(range(H, Lm))
# position dans l'accumulateur combine de chaque colonne de v7 et du bloc
pos_main = [i if i < H else i - H + H + h for i in range(Lm)]
pos_blk = [H + j if j < h else (H + h) + H + (j - h) for j in range(B)]
# l0 : nous [0, N/2), eux [N/2, N). Entrees l1 de v7 : nous [0, H), eux [H, Lm) ; bloc : nous [0, h), eux [h, B)
l0_main = [i if i < H else N // 2 + (i - H) for i in range(Lm)]
l0_blk = [H + j if j < h else N // 2 + H + (j - h) for j in range(B)]

with torch.no_grad():
    cm.input.coalesce(); cm.cond_input.coalesce()
    cm.layer_stacks.coalesce_layer_stacks_inplace(); cm.cond_stacks.coalesce_layer_stacks_inplace()

    def combined_ft(model):
        for fn, fm, fb in zip(model.input.features, cm.input.features, cm.cond_input.features):
            w = torch.zeros_like(fn.weight)
            w[:, pos_main] = fm.weight[:, :Lm]
            w[:, pos_blk] = fb.weight[:, :B]
            w[:, N:] = fm.weight[:, Lm:]                     # PSQT de v7
            fn.weight.copy_(w); fn.zero_virtual_weights()
        b = torch.zeros_like(model.input.bias)
        b[pos_main] = cm.input.bias[:Lm]; b[pos_blk] = cm.cond_input.bias[:B]
        model.input.bias.copy_(b)

    # 1) accumulateur combine + piles de v7
    m1 = NNUEModel(F, ModelConfig(L1=N, L2=cm.layer_stacks.L2, L3=cm.layer_stacks.L3))
    combined_ft(m1)
    ls_src, ls_dst = cm.layer_stacks, m1.layer_stacks
    w = torch.zeros_like(ls_dst.l1.linear.weight)
    w[:, l0_main] = ls_src.l1.linear.weight
    ls_dst.l1.linear.weight.copy_(w); ls_dst.l1.linear.bias.copy_(ls_src.l1.linear.bias)
    ls_dst.l1.factorized_linear.weight.zero_(); ls_dst.l1.factorized_linear.bias.zero_()
    ls_dst.l2.load_state_dict(ls_src.l2.state_dict()); ls_dst.output.load_state_dict(ls_src.output.state_dict())

    # 2) accumulateur combine + piles conditionnelles
    cs = cm.cond_stacks
    m2 = NNUEModel(F, ModelConfig(L1=N, L2=cs.L2, L3=cs.L3))
    combined_ft(m2)
    d2 = m2.layer_stacks
    w = torch.zeros_like(d2.l1.linear.weight)
    w[:, l0_main] = cs.l1.linear.weight
    w[:, l0_blk] = cs.l1_extra.linear.weight
    d2.l1.linear.weight.copy_(w); d2.l1.linear.bias.copy_(cs.l1.linear.bias + cs.l1_extra.linear.bias)
    d2.l1.factorized_linear.weight.zero_(); d2.l1.factorized_linear.bias.zero_()
    d2.l2.load_state_dict(cs.l2.state_dict()); d2.output.load_state_dict(cs.output.state_dict())

for m, p in [(m1, out + ".nnue"), (m2, out + ".cond.nnue")]:
    wr = M.NNUEWriter(m, f"bloc conditionnel {B}, condition {cm.cond_name}", ft_compression="none", verbose=False)
    open(p, "wb").write(wr.buf); print("ecrit", p, len(wr.buf))

if "--check" in sys.argv:
    # Controle en flottant : m1 = modele d'origine hors condition, m2 = modele d'origine sous condition.
    import data_loader
    from model.cond_model import cond_mask
    tsv = sys.argv[sys.argv.index("--check") + 1]
    fens = [l.split("\t")[0] for l, _ in zip(open(tsv), range(4000))]
    bt = data_loader.get_sparse_batch_from_fens(cm.input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
    us, them, wi, bi, o, s, pc = bt.contents.get_tensors("cpu")
    with torch.no_grad():
        ref = cm(us, them, wi, bi, pc, False, False).flatten()
        a = m1(us, them, wi, bi, pc, False, False).flatten()
        b = m2(us, them, wi, bi, pc, False, False).flatten()
        c = cond_mask(wi, cm.cond_name)
        S = cm.quantization.nnue2score
        print("hors condition : ecart max (cp)", ((ref - a)[~c].abs().max() * S).item())
        print("sous condition : ecart max (cp)", ((ref - b)[c].abs().max() * S).item(), "| part sous condition", c.float().mean().item())
