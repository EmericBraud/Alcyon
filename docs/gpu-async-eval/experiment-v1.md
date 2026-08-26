# Expérience v1 : CNN baseline (8 blocs × 96 canaux, sans plans d'attaque)

Statut : terminée, remplacée par v2 (voir commit `07f70bb`, "Add attack
planes and scale up CNN architecture"). Ce document capture la
méthode, les résultats et les apprentissages avant que le code source
(model.py, plane_batch.h/.cpp) ne change pour v2.

## Objectif

Évaluer si un petit CNN, entraîné à approximer l'évaluation Stockfish
et le résultat réel des parties, peut produire un signal utile en
complément de la NNUE existante du moteur — préalable au design
"async GPU eval" décrit dans `architecture.md`.

## Méthode

### Données

- `test80-2024-01-jan-2tb7p.min-v2.v6.binpack` (~9.2 GB), format
  binpack de `nnue-pytorch`, réutilisé tel quel via le submodule
  vendored (`training/nnue-pytorch/`).
- Split train/validation déterministe par hash de position (FNV-1a
  sur la sérialisation 24 octets de `CompressedPosition`), pas de
  découpage physique du fichier (le format est delta-compressé par
  partie).
- Chargeur C++ maison (`training/cnn/data_loader/`) qui construit
  directement des plans denses 8×8×19 depuis
  `binpack::TrainingDataEntry`, sans passer par les extracteurs de
  features NNUE (`HalfKAv2_hmExtractor` etc.) du submodule.

### Entrée (19 plans)

- 12 plans de pièces (nôtres / adverses, orientées côté-au-trait via
  un flip de rangée)
- 1 plan "trait aux blancs"
- 4 plans de droits au roque
- 1 plan en passant (one-hot)
- 1 plan rule50 normalisé

Pas de plans d'attaque, pas d'historique de répétition (le binpack
n'a pas de fenêtre d'historique de partie).

### Architecture

- Stem conv 3×3 (19→96 canaux) + BN + ReLU
- 8 blocs résiduels (conv3×3+BN+ReLU → conv3×3+BN → +skip → ReLU),
  résolution 8×8 conservée de bout en bout, pas de pooling
- **Skip PSQT** : conv 1×1 sur les plans de pièces uniquement, sommée
  sur le plateau, une valeur par bucket de phase — contourne tout le
  trunk, analogue à la table PSQT de la NNUE
- **Skip mid-trunk** : sortie du bloc résiduel médian ET sortie finale
  toutes deux global-average-pooled puis concaténées avant la tête de
  valeur — analogue à la concaténation L1+L2 avant la couche de sortie
  de la NNUE (`nnue_model.hpp`)
- 4 têtes de valeur (MLP 2 couches) sélectionnées par bucket de phase
  (découpage par nombre de pièces hors rois : ≥24 / 16-23 / 8-15 / <8)
- Sortie : un seul logit scalaire de probabilité de victoire (pas de
  tête WDL 3-voies, pas de tête policy)
- **~1,42M paramètres**

### Cible d'entraînement

`binary_cross_entropy_with_logits` contre une cible mélangée par
annealing de lambda :

```
target = lambda * sigmoid(score / SCORE_SCALE) + (1 - lambda) * (result + 1) / 2
```

avec `lambda` décroissant linéairement de 1.0 à 0.3 sur l'entraînement
(intuition "teacher-student" : au début on fait confiance au score de
recherche Stockfish, à la fin on corrige vers le résultat réel des
parties).

### Infrastructure d'entraînement

- Testé d'abord localement sur Mac (MPS), jugé trop lent pour un run
  complet.
- Déplacé sur GPU loué (gpu.ai) : d'abord une RTX 4090, puis (pour la
  reprise/extension) une RTX 5090.
- Bug corrigé en cours de route : sur-souscription de threads CPU —
  PyTorch détectait le nombre de cœurs de l'hôte (255) au lieu du
  quota cgroup réel du conteneur (~30), causant une contention massive
  (load average 117.9, GPU affamé par intermittence à 69-89% SM
  utilization). Fix : argument `--cpu-threads` appelant explicitement
  `torch.set_num_threads()` / `set_num_interop_threads()`. Gain post-fix :
  load average 5.01, GPU SM 93%, +10% de débit (35 500 → 38 900 pos/s).
- RTX 5090 (Blackwell, sm_120) nécessite le wheel PyTorch `cu129` — le
  wheel `cu126` échoue avec `no kernel image is available for
  execution on the device`.
- Support de reprise ajouté (`--resume-from`) pour survivre aux coupures
  d'instance GPU, plus un mode dédié `--extend-steps` pour prolonger un
  entraînement dont le schedule LR/lambda était déjà totalement
  descendu, sans corrompre les courbes (nouveau warmup+decay propre,
  lambda figé au plancher).
- Run complet en deux temps : 100 000 steps initiaux (RTX 4090), puis
  100 000 steps d'extension (RTX 5090, reprise depuis le checkpoint
  100k) → 200 000 steps au total, batch size 4096.

### Méthodologie d'évaluation

Développée dans `training/cnn/eval_compare/`, après plusieurs
itérations pour éliminer des biais :

1. **Static, pas searched** : comparer contre l'eval statique de
   Stockfish (commande UCI `eval`, sans recherche), pas son score
   recherché — sur un test tactique (WAC, 299 positions), un score
   recherché résout des combinaisons qu'aucun évaluateur statique ne
   peut voir, ce qui aurait mesuré la profondeur de recherche plutôt
   que la qualité de l'éval.
2. **Corrélation, pas MAE** : métrique principale car invariante à
   l'échelle — importante puisque le `cp_scale` du CNN n'a jamais été
   calibré nativement (contrairement à la NNUE qui sort des centipions
   par construction).
3. **Calibration d'échelle cross-validée** : refit de l'échelle par
   moindres carrés sur WAC lui-même est circulaire si évalué sur le
   même ensemble → split en 2 folds, chaque fold noté avec l'échelle
   fittée sur l'AUTRE fold.
4. **Calibration binpack rejetée** : fitter l'échelle sur le champ
   `score` du binpack (un score recherché, pas statique) donne une
   échelle ~7× différente de celle fittée sur WAC — cibles de nature
   différente, non transférable.
5. Diagnostic clé : **corrélation entre les erreurs** (pas les
   prédictions) de la NNUE et du CNN. Haute = erreurs redondantes, pas
   de valeur ajoutée à combiner. Basse = zones d'erreur réellement
   différentes → précondition pour que la correction asynchrone ait un
   sens.

## Résultats

### Courbe d'apprentissage (corrélation vs Stockfish static eval, WAC)

| Step | Corrélation CNN | MAE |
|---|---|---|
| ~intermédiaire (session précédente) | ~0.44 → 0.85 (montée progressive) | — |
| 180 000 | 0.8663 | 405.3 |
| 200 000 (final) | 0.8708 | 407.1 |
| NNUE (référence) | **0.9499** | 328.2 |

Progression continue jusqu'à ~180k, puis plateau net entre 180k et
200k (+0.5%, dans le bruit) — cohérent avec la fin de la décroissance
cosine du LR (LR=0 à step 200 000). `val_loss` restait plate (~0.598)
sur les 20 000 derniers steps.

### Analyse par position (step 200 000, échelle recalibrée par CV)

- Échelle optimale recalibrée : **~89** (contre 410 par défaut — le
  score brut était donc massivement sous-exploité à l'échelle initiale)
- Taux de victoire du CNN vs NNUE (erreur la plus faible vs eval
  statique Stockfish), à l'échelle recalibrée : **88.5%** (208/235
  positions WAC valides)
- Corrélation entre les magnitudes d'erreur NNUE/CNN : **0.23** (basse)
- Les plus gros écarts en faveur du CNN sont systématiquement des cas
  où la NNUE surestime massivement l'avantage matériel (+1000 à +1900
  cp d'erreur) sur des positions tactiquement chargées, alors que le
  CNN reste proche de l'eval statique Stockfish (souvent <100 cp
  d'écart) — piste : la NNUE sur-évalue en présence de tension
  tactique non résolue par une éval statique seule.

## Apprentissages

1. **Le taux de victoire brut ("43%") était trompeur** — dû à une
   échelle de sortie non calibrée, pas à une infériorité réelle du
   modèle. Toujours vérifier un artefact d'échelle avant de conclure
   à une différence de qualité entre deux évaluateurs à sorties non
   nativement comparables.
2. **La corrélation d'erreur, pas juste la performance individuelle,
   est le bon diagnostic** pour juger si un second modèle apporte un
   signal complémentaire — deux modèles également précis mais
   corrélés dans leurs erreurs n'apportent rien en combinaison.
3. **Le plateau à 180k-200k steps, avec LR déjà à 0, ne veut pas dire
   "modèle au maximum de son potentiel"** — c'est une limite du
   schedule d'entraînement (fin de la décroissance cosine), pas
   nécessairement une limite de capacité du modèle. Prolonger avec le
   même schedule ne sert à rien (LR=0) ; un nouveau cycle
   `--extend-steps` donne un gain marginal, cohérent avec des
   rendements décroissants sur cette taille de modèle et cet ensemble
   de plans d'entrée.
4. **Le modèle (1,42M paramètres, 19 plans bruts) était probablement
   sous-dimensionné et sous-informé** relativement à la NNUE de
   référence : un CNN doit "redécouvrir" la portée d'une tour ou d'un
   fou à travers plusieurs couches de convolution 3×3, alors que cette
   information est trivialement calculable en amont (bitboards
   d'attaque déjà présents dans le moteur C++). Décision (voir v2) :
   ajouter des plans d'attaque en entrée et augmenter la taille du
   trunk (14 blocs × 160 canaux, ~6,9M paramètres) plutôt que
   continuer à entraîner cette architecture.
5. **La sur-souscription de threads CPU dans un environnement
   conteneurisé loué est un piège silencieux** — `nproc`/
   `torch.get_num_threads()` par défaut rapportent le nombre de cœurs
   de l'hôte, pas le quota cgroup réel alloué au conteneur ; vérifier
   `/sys/fs/cgroup/cpu.max` avant de faire confiance aux valeurs par
   défaut de PyTorch sur ce type d'infrastructure.
6. **Toujours faire un essai à petite échelle (`--steps` court) avant
   un lancement long sans supervision** — a permis de détecter un bug
   de chemin relatif dans le chargeur ctypes avant un run de nuit.

## Suite (v2)

Voir le commit `07f70bb` : ajout de 2 plans d'attaque binaires
(cases attaquées par nous / par eux), et passage à 14 blocs × 160
canaux avec un module d'attention par canal (squeeze-and-excitation)
dans chaque bloc résiduel, comme dans le trunk de Leela Chess Zero.
Entraînement relancé from scratch (l'architecture n'est pas compatible
avec les checkpoints v1).
