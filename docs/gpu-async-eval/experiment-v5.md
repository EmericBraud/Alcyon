# Expérience v5 : correction résiduelle additive de la NNUE

Statut : terminée. Ce document capture la méthode, les résultats et
les apprentissages du premier run complet de v5 (voir
`v5-hybrid-nnue-cnn.md` pour la conception détaillée).

## Objectif

Tester l'hypothèse : plutôt qu'entraîner un CNN comme évaluateur
autonome (v1-v4), entraîner un CNN à corriger l'erreur résiduelle de
la NNUE existante du moteur (`logit_final = trunk_logits +
nnue_logit`, `nnue_logit` étant une constante non-entraînable). Objectif
secondaire : mesurer si cette approche permet d'atteindre, à capacité
égale (même taille de trunk que v1), une meilleure précision qu'un
évaluateur autonome.

## Méthode

### Architecture

- Trunk identique en taille à v1 : 8 blocs résiduels × 96 canaux avec
  squeeze-and-excitation, 33 plans d'entrée (les 19 plans de v1 + 6
  plans d'attaque par type de pièce + 2 plans de distance au roi
  introduits en v3/v4), **~1,5M paramètres**.
- Pas de skip PSQT (redondant avec le PSQT déjà appris par la NNUE,
  contenu dans `nnue_logit`).
- 8 têtes de valeur par bucket de phase (mêmes limites que v4 :
  ≥26/22-25/18-21/14-17/10-13/6-9/3-5/<3 pièces hors rois).
- `nnue_logit` calculé une fois par position dans le chargeur C++
  (bridge isolé vers le moteur NNUE réel du projet, voir
  `nnue_bridge.cpp`), converti d'un score NNUE brut en logit via une
  courbe de calibration monotone par morceaux (64 bins, fit par
  quantiles + isotonic pooling) — un scalaire d'échelle unique s'est
  révélé insuffisant (échelle effective variant ~1,7× entre positions
  quasi-égales et décisives).

### Infrastructure

- Serveur loué 4× RTX 4080 (PCIe, pas de NVLink), entraînement
  distribué DDP (`torchrun --nproc_per_node=4 --standalone`).
- Batch size 16384/GPU (65536 global) après diagnostic — 4096/GPU
  initial ne saturait le GPU qu'à ~50% (coût de synchronisation NCCL
  disproportionné pour un modèle aussi petit), 16384/GPU a porté
  l'utilisation à 85-100%.
- 25000 steps (réduit de 100000 initialement prévus, pour conserver le
  même budget total de données — ~1,64 milliard de positions — après
  avoir quadruplé la taille de batch), LR 2e-3 (règle racine carrée
  pour Adam vu le batch global ×4).
- Débit stable ~190-194k pos/s, run complet en **~144 minutes**.

### Méthodologie d'évaluation

Contrairement à v1-v4 (comparaison contre l'éval statique Stockfish
sur WAC.epd, 299 positions tactiques), la comparaison v5 utilise le
split de validation du binpack lui-même (hash-based, ~49k positions),
contre deux cibles :

1. **Score Stockfish recherché** (champ `score` natif du binpack).
2. **Résultat réel de la partie (WDL)**, via
   `corr(tanh(logit / SCORE_SCALE), result)`.

Ce choix corrige un biais identifié en cours de route : l'éval
statique Stockfish sur seulement 235 positions tactiques n'est pas
représentatif de l'usage réel (accuracy vs résultat de partie), et
sous-estimait fortement le gain réel de la correction (au même stade
d'entraînement, la corrélation WAC montrait le CNN pire que la NNUE
seule, alors que la corrélation WDL/score-recherché sur le vrai split
de validation montrait déjà un gain net). Script :
`eval_compare/compare_vs_binpack.py`.

## Résultats

### Courbe d'apprentissage (corrélation vs score recherché / WDL, split validation)

| Step | vs score recherché | vs WDL |
|---|---|---|
| 5000 (20%) | — (WAC seul : pire que NNUE) | — |
| 7500 (30%) | dépasse déjà NNUE seul | dépasse déjà NNUE seul |
| 16250 (65%) | 0.7158 | 0.4438 |
| 18750 (75%) | 0.7065 | 0.4508 |
| **25000 (final)** | **0.7242** | **0.4626** |

### Comparaison finale, toutes générations (split validation, ~49k positions)

| Modèle | Taille | vs score recherché | vs WDL |
|---|---|---|---|
| NNUE seule | — | 0.6810 | 0.4199 |
| v1 (final, 200k steps) | 8×96 | 0.7073 | 0.4556 |
| **v5 (final, 25k steps)** | 8×96 + NNUE | **0.7242** | **0.4626** |
| v4 (final, 400k steps) | 20×224 | 0.7561 | 0.4806 |

## Apprentissages

1. **L'hypothèse de départ est validée à cette échelle** : v5 bat NNUE
   seule sur les deux métriques (+0.043 score, +0.043 WDL), et bat
   aussi v1 malgré une taille de trunk strictement identique (8×96) —
   partir d'un signal NNUE déjà correct et n'apprendre que le résidu
   est une tâche plus facile qu'apprendre une évaluation complète
   from scratch, à capacité de modèle égale.
2. **v4 reste devant** (0.7561/0.4806), mais c'est un modèle ~5× plus
   gros (20×224 vs 8×96) — pas une comparaison à coût d'inférence égal.
   Question ouverte pour une suite : un trunk v5 agrandi vers la
   taille de v4 dépasserait-il v4 tout en restant moins cher qu'un
   évaluateur autonome équivalent ?
3. **Le choix de la cible d'évaluation change la conclusion** : au
   même checkpoint (step 5000, 20%), la méthodologie WAC/static
   concluait que le CNN dégradait la précision (MAE 774 vs 328 pour
   NNUE), alors que la méthodologie score-recherché/WDL sur le split
   de validation réel montrait déjà (dès step 7500) un gain net. Le
   WAC reste utile pour l'analyse de points faibles tactiques
   spécifiques, mais n'est pas le bon test pour juger si une
   correction améliore la précision réelle.
4. **Le calibrage non-linéaire du score NNUE était nécessaire** : un
   scalaire d'échelle unique (`NNUE_SCALE`) était insuffisant — l'échelle
   effective NNUE-vers-logit variait ~1,7× selon la magnitude du score,
   d'où la courbe de calibration par morceaux.
5. **Batch size et NCCL** : pour un petit modèle (1,5M paramètres) sur
   du matériel multi-GPU sans NVLink, le coût de synchronisation fixe
   par step domine à petit batch — un batch 4× plus gros par GPU a
   porté l'utilisation GPU de ~50% à 85-100% et le débit de ~148k à
   ~194k pos/s.

## Suite

Piste envisagée : un modèle MoE (mixture of experts) avec gating
appris end-to-end, sur l'hypothèse que l'erreur résiduelle de la NNUE
a une structure en clusters (types de positions récurrents mal
évalués) plutôt que d'être un bruit homogène — à explorer une fois
cette hypothèse vérifiée empiriquement sur les résidus de ce
checkpoint.
