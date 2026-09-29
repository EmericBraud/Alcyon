"""Ordonnancement appris des coups calmes : combien de gaspillage un reseau retire-t-il ?

    python3 order_fit.py <dossier de dumps game_NNN.bin (+ .l0)> <probas de tirage> [epochs]

Dumps : ALCYON_ORDER_DUMP (search::OrderRecord), aux noeuds ou un coup de
l'etape QUIETS coupe : tous les coups calmes de l'etape, et l'l0 du noeud.
Parties 0-219 entrainement, 220-259 validation, 260+ test.

Gaspillage d'un noeud = noeuds des coups calmes cherches avant celui qui coupe.
Avec un autre ordre, on compte les coups classes avant le coup qui coupe :
leur sous-arbre s'ils ont ete cherches, sinon le cout moyen d'un coup calme
qui echoue a cette profondeur (appris en entrainement). Pondere par 1 / p du
tirage des noeuds. Compare : ordre actuel (score d'ordonnancement), et le
reseau (l0 du noeud -> 32, puis un score par coup a partir des features du
coup et du noeud), entraine en softmax sur les coups du noeud.
"""
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn

I64 = ["node", "subtree"]
I32 = ("depth ply cut_node improving eval_beta pos picked searched is_cut order_score history cont1 cont2 "
       "piece from_ to gives_check n_quiets halfmove pieces").split()
DTYPE = np.dtype([(n, "<i8") for n in I64] + [(n, "<i4") for n in I32])
assert DTYPE.itemsize == 96
EVAL_NONE = 1 << 30


def load(folder):
    sets = ([], [], [])
    for path in sorted(glob.glob(os.path.join(folder, "game_*.bin"))):
        g = int(os.path.basename(path)[5:8])
        r = np.fromfile(path, dtype=DTYPE)
        l0 = np.fromfile(path + ".l0", dtype=np.uint8).reshape(-1, 1024)
        nodes = np.unique(r["node"])
        n = min(len(nodes), len(l0))
        r = r[r["node"] < n]  # un noeud sans son l0 (fin de fichier coupee) est retire
        sets[0 if g < 220 else 1 if g < 260 else 2].append((g, r, l0[:n]))
    out = []
    for parts in sets:
        rs, l0s, off = [], [], 0
        for g, r, l0 in parts:
            r = r.copy()
            r["node"] += off  # id globaux : l0[node] est l'l0 du noeud
            off += len(l0)
            rs.append(r)
            l0s.append(l0)
        out.append((np.concatenate(rs), np.concatenate(l0s)))
    return out


def signlog(a):
    a = a.astype(np.float64)
    return np.sign(a) * np.log1p(np.abs(a))


def move_features(r):
    eb = r["eval_beta"].astype(np.float64)
    known = eb != EVAL_NONE
    eb = np.where(known, np.clip(eb, -1500, 1500) / 200, 0)
    cols = [signlog(r["order_score"]), signlog(r["history"]), signlog(r["cont1"]), signlog(r["cont2"]),
            r["gives_check"], r["depth"] / 10, r["ply"] / 10, r["cut_node"], r["improving"], eb, known,
            r["halfmove"] / 50, r["pieces"] / 32, np.log1p(r["n_quiets"])]
    x = np.column_stack(cols).astype(np.float32)
    return x


class Net(nn.Module):
    def __init__(self, n_x, hidden=32):
        super().__init__()
        self.l0 = nn.Linear(1024, hidden)
        self.piece = nn.Embedding(8, 8)
        self.sq_from = nn.Embedding(64, 8)
        self.sq_to = nn.Embedding(64, 8)
        self.head = nn.Sequential(nn.Linear(hidden + n_x + 24, 64), nn.ReLU(), nn.Linear(64, 1))
        self.base = nn.Linear(n_x, 1)  # partie lineaire (le score d'ordonnancement y passe directement)

    def forward(self, l0_node, x, piece, fr, to):
        h = torch.relu(self.l0(l0_node.float() / 127.0))
        e = torch.cat([self.piece(piece), self.sq_from(fr), self.sq_to(to)], 1)
        return self.base(x).squeeze(1) + self.head(torch.cat([h, x, e], 1)).squeeze(1)


def node_weights(r, probs):
    p = [float(v) for v in probs.split(",")]
    table = np.array([1.0] + [p[min(d, len(p)) - 1] for d in range(1, 64)])
    return 1.0 / table[np.minimum(r["depth"], 63)]


def waste(r, score, fail_cost, w):
    """Gaspillage pondere si l'on range chaque noeud par score decroissant."""
    order = np.lexsort((-score, r["node"]))  # par noeud, score decroissant
    rs, nodes = r[order], r["node"][order]
    starts = np.r_[0, np.nonzero(np.diff(nodes))[0] + 1]
    # cherche : son sous-arbre ; tire mais elague (LMP, futility, illegal) : 0 ;
    # jamais tire (apres la coupure) : cout moyen d'un coup calme qui echoue.
    cost = np.where(rs["searched"] == 1, rs["subtree"],
                    np.where(rs["picked"] == 1, 0, fail_cost[np.minimum(rs["depth"], 63)]))
    cost = np.where(rs["is_cut"] == 1, 0, cost).astype(np.float64)
    cum = np.cumsum(cost)
    # cout avant le coup qui coupe, dans chaque noeud
    cut_idx = np.nonzero(rs["is_cut"] == 1)[0]
    node_start = starts[np.searchsorted(starts, cut_idx, side="right") - 1]
    before = cum[cut_idx] - cost[cut_idx] - np.where(node_start > 0, cum[node_start - 1], 0)
    return float((before * w[order][cut_idx]).sum())


def main():
    folder, probs = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", os.cpu_count())))
    (tr, l0tr), (va, l0va), (te, l0te) = load(folder)
    print(f"noeuds train {len(l0tr)}, val {len(l0va)}, test {len(l0te)} ; coups calmes par noeud {len(tr) / len(l0tr):.1f}")

    # Cout moyen d'un coup calme cherche qui echoue, par profondeur (train).
    m = (tr["searched"] == 1) & (tr["is_cut"] == 0)
    d = np.minimum(tr["depth"][m], 63)
    fail_cost = (np.bincount(d, weights=tr["subtree"][m], minlength=64) + 1) / (np.bincount(d, minlength=64) + 1)

    def tensors(r, l0):
        return (torch.from_numpy(np.ascontiguousarray(l0)), torch.from_numpy(move_features(r)),
                torch.from_numpy(np.minimum(r["piece"], 7).astype(np.int64)), torch.from_numpy(r["from_"].astype(np.int64)),
                torch.from_numpy(r["to"].astype(np.int64)), torch.from_numpy(r["node"]))

    TR, VA, TE = tensors(tr, l0tr), tensors(va, l0va), tensors(te, l0te)
    xm, xs = TR[1].mean(0), TR[1].std(0) + 1e-6
    net = Net(TR[1].shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    def scores(T):
        with torch.no_grad():
            out = []
            for i in range(0, len(T[1]), 65536):
                sl = slice(i, i + 65536)
                out.append(net(T[0][T[5][sl]], (T[1][sl] - xm) / xs, T[2][sl], T[3][sl], T[4][sl]))
            return torch.cat(out).numpy()

    wva, wte = node_weights(va, probs), node_weights(te, probs)
    base_va = waste(va, -va["pos"].astype(np.float64), fail_cost, wva)
    base_te = waste(te, -te["pos"].astype(np.float64), fail_cost, wte)
    cut_tr = torch.from_numpy(tr["is_cut"] == 1)
    node_tr = TR[5]
    n_nodes = len(l0tr)
    best, best_state = None, None
    bs_nodes = 4096
    # coups groupes par noeud (contigus) : lots de noeuds entiers pour la softmax
    starts = np.r_[0, np.nonzero(np.diff(tr["node"]))[0] + 1, len(tr)]
    for ep in range(epochs):
        perm = np.random.permutation(n_nodes)
        tot = 0.0
        for i in range(0, n_nodes, bs_nodes):
            nb = np.sort(perm[i:i + bs_nodes])
            idx = np.concatenate([np.arange(starts[k], starts[k + 1]) for k in nb])
            it = torch.from_numpy(idx)
            z = net(TR[0][node_tr[it]], (TR[1][it] - xm) / xs, TR[2][it], TR[3][it], TR[4][it])
            # softmax par noeud : log-somme-exp par groupe
            grp = torch.from_numpy(np.repeat(np.arange(len(nb)), [starts[k + 1] - starts[k] for k in nb]))
            zmax = torch.zeros(len(nb)).index_reduce_(0, grp, z.detach(), "amax", include_self=False)
            lse = torch.log(torch.zeros(len(nb)).index_add_(0, grp, torch.exp(z - zmax[grp]))) + zmax
            loss = (lse - z[cut_tr[it]]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(nb)
        sched.step()
        wv = waste(va, scores(VA), fail_cost, wva)
        mark = ""
        if best is None or wv < best:
            best, best_state, mark = wv, {k: v.clone() for k, v in net.state_dict().items()}, "  *"
        print(f"  epoch {ep + 1}/{epochs} : loss {tot / n_nodes:.4f} ; gaspillage val {wv / base_va - 1:+.1%} vs ordre actuel{mark}", flush=True)
    net.load_state_dict(best_state)
    wt = waste(te, scores(TE), fail_cost, wte)
    print(f"\ntest : gaspillage dans l'etape QUIETS {wt / base_te - 1:+.1%} vs ordre actuel "
          f"(ordre parfait : -100 %) ; gaspillage actuel {base_te:.3g} noeuds ponderes")
    torch.save({"state": net.state_dict(), "xm": xm, "xs": xs, "fail_cost": fail_cost}, os.path.join(folder, "order_net.pt"))


if __name__ == "__main__":
    main()
