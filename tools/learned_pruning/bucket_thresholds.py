"""Seuils par expert (bucket) pour un modele de train_safe.py.

    python3 bucket_thresholds.py <test.bin> <R> <modele.pt> [--err count|cost]

On cherche l'economie maximale pour un budget d'erreurs. Le lagrangien se
decompose par bucket : pour un prix lambda d'une erreur, chaque expert
choisit SEUL le seuil qui maximise economie - lambda * erreurs. En balayant
lambda on trace la courbe economie / erreurs, comparee a un seuil unique.

Seuils choisis sur une moitie du test, evalues sur l'autre (tirage des
noeuds) : sinon le resultat serait optimiste. --err cost pondere chaque
erreur par le cout complet du noeud (proxy du "danger").
"""
import sys

import numpy as np

import moe_diag
import train_mlp
import train_safe

N_CAND = 300  # seuils candidats par expert : quantiles de p dans le bucket


def candidates(p, bucket, n_b):
    """Seuils candidats de chaque expert : quantiles de ses predictions (fins
    pres de 1, ou se concentre un bon modele), plus "n'accepte rien"."""
    q = 1 - np.logspace(0, -4, N_CAND)  # 0 .. 0.9999, dense vers le haut
    return np.stack([np.append(np.quantile(p[bucket == b], q) if (bucket == b).any() else np.ones(N_CAND), 2.0)
                     for b in range(n_b)])


def bucket_stats(p, wrong, saved, err_w, bucket, T):
    """Economie et erreurs de chaque expert pour chacun de ses seuils T[b]."""
    n_b, n_t = T.shape
    S, E = np.zeros((n_b, n_t)), np.zeros((n_b, n_t))
    for b in range(n_b):
        s = bucket == b
        pb, sb, eb = p[s], saved[s], wrong[s] * err_w[s]
        order = np.argsort(-pb)
        ps, cs, ce = pb[order], np.concatenate([[0], np.cumsum(sb[order])]), np.concatenate([[0], np.cumsum(eb[order])])
        n_acc = np.searchsorted(-ps, -T[b], side="right")  # nombre de noeuds avec p >= t
        S[b], E[b] = cs[n_acc], ce[n_acc]
    return S, E


def choose(S, E, lam):
    """Seuil de chaque bucket qui maximise S - lam * E (independamment)."""
    return np.argmax(S - lam * E, axis=1)


def main():
    test, R = sys.argv[1], int(sys.argv[2])
    model = sys.argv[3]
    err_mode = sys.argv[5] if len(sys.argv) > 5 and sys.argv[4] == "--err" else "count"
    k = R - 1
    r, l0, idx = train_mlp.load(test)
    p, y = moe_diag.predict(model, r, l0, idx, k, None, None)
    rt = r[idx]
    red_fh = rt["red_score"][:, k] >= rt["beta"]
    wrong = y < 0.5
    saved = np.maximum(rt["subtree"] - rt["red_nodes"].sum(1), 1).astype(np.float64)
    err_w = saved if err_mode == "cost" else np.ones(len(p))
    bucket, n_b = train_safe.buckets(rt, red_fh, "ds")

    half = np.random.default_rng(3).random(len(p)) < 0.5
    tot_saved, tot_err = saved[~half].sum(), (err_w[~half]).sum()
    T = candidates(p[half], bucket[half], n_b)
    S_a, E_a = bucket_stats(p[half], wrong[half], saved[half], err_w[half], bucket[half], T)
    S_b, E_b = bucket_stats(p[~half], wrong[~half], saved[~half], err_w[~half], bucket[~half], T)

    # Seuil unique : meme valeur pour tous les experts.
    G = np.tile(np.append(1 - np.logspace(0, -5, 600), 2.0), (n_b, 1))
    S_g, E_g = bucket_stats(p[~half], wrong[~half], saved[~half], err_w[~half], bucket[~half], G)
    glob = [(S_g[:, j].sum() / tot_saved, E_g[:, j].sum() / tot_err) for j in range(G.shape[1])]
    # Par expert : lambda balaye, seuils choisis sur la moitie a.
    per = []
    ratio = S_a.sum() / max(E_a.sum(), 1e-9)
    for lam in ratio * np.logspace(-3, 4, 3000):
        j = choose(S_a, E_a, lam)
        per.append((S_b[np.arange(n_b), j].sum() / tot_saved, E_b[np.arange(n_b), j].sum() / tot_err, lam, j))

    def err_at(curve, target):
        ok = [e for s, e, *_ in curve if s >= target]
        return min(ok) if ok else float("nan")

    unit = "des erreurs ponderees par le cout" if err_mode == "cost" else "des noeuds en erreur"
    print(f"R={R}, {model.split('/')[-1]}, erreurs = {err_mode} ; evalue sur la moitie tenue a l'ecart")
    print(f"erreurs non detectees (part {unit}) a economie de 10 / 20 / 30 / 40 / 50 % du cout complet")
    for name, c in (("seuil unique", glob), ("par expert", per)):
        vals = [err_at(c, t) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]
        print(f"  {name:13s} " + "  ".join(f"{v:8.4%}" for v in vals))

    # Seuils retenus au point de fonctionnement a 30 % d'economie.
    best = min((c for c in per if c[0] >= 0.3), key=lambda c: c[1], default=None)
    if best is not None:
        print(f"\nseuils par expert a ~30 % d'economie (lambda = {best[2]:.3g}) :")
        for b in range(n_b):
            s = bucket == b
            t = T[b, best[3][b]]
            print(f"  depth {moe_diag.DEPTH_NAMES[b // 2]:>3s} {'FH' if b % 2 else 'FL'} : "
                  f"seuil {t:.5f}, accepte {np.mean(p[s] >= t):6.1%} des noeuds du bucket")


if __name__ == "__main__":
    main()
