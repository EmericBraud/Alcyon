"""Evals de plusieurs modeles (.pt, ou base.pt@poids.ckpt) sur les positions calmes d'un tsv
(fen, score, resultat, coup, ply : sortie de dump). A lancer depuis la racine de np.
Usage : eval_fens.py <val.tsv> <sortie.tsv> nom=modele [nom=modele ...]"""
import sys, torch, chess
import data_loader
src, out = sys.argv[1], sys.argv[2]
specs = [a.split("=", 1) for a in sys.argv[3:]]

rows = []
for l in open(src):
    fen, sc, res, mv, ply = l.rstrip("\n").split("\t"); sc = int(sc)
    if sc == 32002 or abs(sc) >= 10000: continue
    b = chess.Board(fen); m = chess.Move.from_uci(mv)
    if b.is_check() or b.is_capture(m) or m.promotion: continue
    rows.append((fen, sc, int(res), int(ply)))
print("positions", len(rows), flush=True)

def load(spec):
    if "@" in spec:
        base, ck = spec.split("@")
        nnue = torch.load(base, weights_only=False, map_location="cpu")
        sd = torch.load(ck, weights_only=False, map_location="cpu")["state_dict"]
        nnue.load_state_dict(sd)
    else:
        nnue = torch.load(spec, weights_only=False, map_location="cpu")
    m = nnue.model.cuda().eval()
    return m

res = {}
for name, spec in specs:
    m = load(spec); ev = []
    with torch.no_grad():
        for i in range(0, len(rows), 8192):
            fens = [r[0] for r in rows[i:i + 8192]]
            bt = data_loader.get_sparse_batch_from_fens(m.input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
            us, them, wi, bi, outcome, score, pc = bt.contents.get_tensors("cuda")
            v = m.forward(us, them, wi, bi, pc, fake_quantize_acts=False, fake_quantize_weights=False) * m.quantization.nnue2score
            ev += [int(round(x)) for x in v.flatten().tolist()]
            data_loader.destroy_sparse_batch(bt)
    res[name] = ev; print(name, "fait", flush=True)
with open(out, "w") as f:
    for i, r in enumerate(rows):
        f.write("\t".join(map(str, r)) + "\t" + "\t".join(str(res[n][i]) for n, _ in specs) + "\n")
print("ecrit", out)
