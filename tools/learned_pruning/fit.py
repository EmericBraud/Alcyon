"""Etape 1 de docs/learned-pruning.md : fits hors ligne sur les features scalaires.

    python3 fit.py train.bin test.bin

Les .bin viennent du dump du moteur (ALCYON_PRUNE_DUMP, build
-DENABLE_SEARCH_EXPERIMENTS=ON). Train et test doivent venir de POSITIONS
differentes : les noeuds d'une meme recherche sont correles.

Modele : P(fail-high) = sigmoid((m + biais(x)) / sigma(x)), m = eval - beta.
Avec biais et 1/sigma lineaires en x, c'est une regression logistique
ordinaire sur [m, x, m*x] -- convexe, donc un mauvais resultat veut dire que
l'information manque, pas que l'optimisation a rate.

  (a) m seul          : biais, sigma constants par tranche de profondeur
                        (ce que RFP / razoring savent deja)
  (b) m, x, m*x       : biais et sigma dependent des features scalaires
  (b-tt)              : (b) sans les features TT -- ce que la position seule
                        apporte, sans l'information de recherche deja payee

Critere : a precision donnee, quelle part du sous-arbre peut-on couper ?
"""
import multiprocessing
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Miroir de search::PruneRecord (worker.hpp). 112 octets.
DTYPE = np.dtype([
    ("subtree", "<i8"),
    ("beta", "<i4"), ("static_eval", "<i4"), ("eval_prev2", "<i4"),
    ("tt_found", "<i4"), ("tt_score", "<i4"), ("tt_depth", "<i4"), ("tt_flag", "<i4"),
    ("depth", "<i4"), ("ply", "<i4"), ("outcome", "<i4"),
    ("cut_node", "<i4"), ("allow_null", "<i4"), ("halfmove", "<i4"), ("stm", "<i4"),
    ("prev_from_piece", "<i4"), ("prev_to_piece", "<i4"),
    ("us", "<i4", 5), ("them", "<i4", 5),
])
assert DTYPE.itemsize == 112

K_EVAL_NONE = 1 << 30
NO_PIECE = 6
TT_FLAGS = 3  # TT_EXACT, TT_ALPHA, TT_BETA -- one-hot, masque par tt_found
SLICES = [(d, d) for d in range(1, 10)] + [(10, 99)]
PRECISIONS = [0.95, 0.98, 0.99]
PIECE_CP = np.array([100, 300, 300, 500, 900])
MAX_TRAIN = 2_000_000


def load(path):
    r = np.fromfile(path, dtype=DTYPE)
    # Fenetres de mat : hors perimetre (exclues du reseau, cf. le doc).
    ok = (np.abs(r["beta"]) < 9000) & (np.abs(r["static_eval"]) < 9000)
    return r[ok]


def features(r):
    """m (en pions) et les scalaires x. Aucune feature ne contient beta."""
    m = np.clip(r["static_eval"] - r["beta"], -1500, 1500) / 100.0
    found = r["tt_found"].astype(bool)
    prev2_ok = r["eval_prev2"] != K_EVAL_NONE
    # eval_prev2 est l'eval de l'ancetre ply-2, meme camp au trait.
    delta2 = np.where(prev2_ok, np.clip(r["static_eval"] - r["eval_prev2"], -1500, 1500), 0) / 100.0
    tt_delta = np.where(found, np.clip(r["tt_score"] - r["static_eval"], -1500, 1500), 0) / 100.0
    tt_ddepth = np.where(found, np.clip(r["tt_depth"] - r["depth"], -10, 10), 0)
    cols = [
        prev2_ok, delta2,
        found, tt_delta, tt_ddepth,
        *[found & (r["tt_flag"] == f) for f in range(TT_FLAGS)],
        r["cut_node"], r["allow_null"],
        np.minimum(r["halfmove"], 100) / 100.0,
        r["stm"], r["ply"] / 10.0, r["depth"],
        r["prev_to_piece"] != NO_PIECE,
        *[r["prev_from_piece"] == p for p in range(6)],
        *[r["prev_to_piece"] == p for p in range(5)],
        *[r["us"][:, p] for p in range(5)],
        *[r["them"][:, p] for p in range(5)],
        (r["us"] @ PIECE_CP - r["them"] @ PIECE_CP) / 100.0,
    ]
    x = np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)
    return m.astype(np.float32), x


TT_COLS = [2, 3, 4, 5, 6, 7]  # found, tt_delta, tt_ddepth, 3 flags : voir features()


def design(m, x, model):
    if model == "a":
        return m[:, None]
    if model == "b-tt":
        x = np.delete(x, TT_COLS, axis=1)
    return np.concatenate([m[:, None], x, m[:, None] * x], axis=1)


def coverage(p, y, subtree, precision):
    """Part du sous-arbre coupable en gardant la precision cumulee >= cible.

    On coupe les noeuds par confiance decroissante (max(p, 1-p)), dans le
    sens predit ; on s'arrete au plus grand prefixe dont la precision reste
    >= cible. Renvoie (part des noeuds, part du sous-arbre, seuil T)."""
    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf, kind="stable")
    correct = ((p >= 0.5) == y)[order]
    prec = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    ok = np.nonzero(prec >= precision)[0]
    if len(ok) == 0:
        return 0.0, 0.0, 1.0
    k = ok[-1] + 1
    return k / len(p), subtree[order][:k].sum() / subtree.sum(), conf[order][k - 1]


MODELS = ("a", "b-tt", "b")
_TR = _TE = None  # partages avec les workers par fork, pas copies


def fit_one(job):
    """Un fit (tranche, modele). Renvoie la ligne a afficher."""
    (lo, hi), model = job
    rtr = _TR[(_TR["depth"] >= lo) & (_TR["depth"] <= hi)]
    rte = _TE[(_TE["depth"] >= lo) & (_TE["depth"] <= hi)]
    if len(rtr) > MAX_TRAIN:  # ponytail: sous-echantillon, lbfgs reste en minutes
        rtr = rtr[np.random.default_rng(0).choice(len(rtr), MAX_TRAIN, replace=False)]
    ytr, yte = rtr["outcome"] != 1, rte["outcome"] != 1  # NMP compte comme fail-high
    mtr, xtr = features(rtr)
    mte, xte = features(rte)
    scaler = StandardScaler().fit(design(mtr, xtr, model))
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(scaler.transform(design(mtr, xtr, model)), ytr)
    p = clf.predict_proba(scaler.transform(design(mte, xte, model)))[:, 1]
    cov = "  ".join(
        f"prec>={t:.0%}: noeuds {n:5.1%} sous-arbre {st:5.1%} (T={c:.3f})"
        for t in PRECISIONS for n, st, c in [coverage(p, yte, rte["subtree"], t)])
    line = f"  ({model:4s}) logloss {log_loss(yte, p):.4f}  auc {roc_auc_score(yte, p):.4f}  {cov}"
    if model != MODELS[0]:
        return line
    name = f"depth {lo}" if lo == hi else f"depth {lo}-{hi if hi < 99 else '+'}"
    return (f"\n== {name} : tt trouvee {rte['tt_found'].mean():.0%}, train {len(rtr)}, "
            f"test {len(rte)}, fail-high {yte.mean():.1%}\n{line}")


def main(train_path, test_path):
    global _TR, _TE
    _TR, _TE = load(train_path), load(test_path)
    print(f"train {len(_TR)} noeuds, test {len(_TE)} noeuds", flush=True)
    jobs = [(sl, m) for sl in SLICES for m in MODELS]
    # fork : les workers heritent de _TR/_TE sans les relire ni les copier.
    with multiprocessing.get_context("fork").Pool(min(len(jobs), os.cpu_count())) as pool:
        for line in pool.imap(fit_one, jobs):  # imap garde l'ordre d'affichage
            print(line, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
