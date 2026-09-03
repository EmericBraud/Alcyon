# Mesures : optimisation du débit GPU-eval, choix v3 vs v6, et design consultatif

Ce document rassemble les mesures effectuées lors de la session qui a
suivi l'implémentation initiale du sous-système GPU-eval (voir
`architecture.md`) : optimisation du pipeline de soumission, choix du
modèle CNN à utiliser, et validation empirique du design consultatif
NNUE/CNN. Chaque section correspond à une question testée, avec le
résultat mesuré, pas une supposition.

## 1. Débit GPU réel vs capacité GPU

Commande de mesure ajoutée : `gpubench` (UCI), isole `infer_batch()` de
la recherche/encodage.

| batch | positions/s |
|---|---|
| 1 | ~60-210 |
| 8 | ~1100-1400 |
| 32 | ~2500-2700 |
| 40 (config initiale) | ~2400-2900 |
| 64 | ~3000-3100 |
| 128-256 (plateau) | **~3200-3400** |

**Plafond GPU mesuré : ~3300 positions/s** en régime saturé, sur ce
réseau (14 blocs SE, 160 canaux).

Débit de soumission réel de l'engin (avant optimisations) : ~450-500
positions/s (mesuré via `gpuevalstats`), soit ~15% de la capacité GPU
— le goulot n'était pas le calcul GPU mais le nombre de positions
soumises côté recherche.

### Tentatives d'optimisation du calcul GPU lui-même (abandonnées)

- **float16** (poids + calcul) : casse le runtime Metal sur ce
  driver — `runWithFeeds:` se bloque indéfiniment dès qu'une nouvelle
  taille de batch apparaît, puis crash (`failed assertion`). Reverté.
- **`MPSGraphExecutable` précompilé** (au lieu de `runWithFeeds:` à
  chaque appel) : plante avec une forme de batch dynamique
  (`all ndarrays should have been allocated`), et même corrigé,
  restait instable (latence croissante entre appels, symptôme de
  recompilations internes répétées). Reverté.
- Conclusion : le calcul GPU lui-même n'était de toute façon pas le
  facteur limitant (voir ci-dessus), donc l'effort a été redirigé vers
  le débit de soumission plutôt que la vitesse de calcul brute.

## 2. Augmentation du débit de soumission

- **Bug trouvé et corrigé** : le throttle du hook chaud utilisait
  `local_nodes` (compteur qui se remet à 0 tous les 32768 nœuds), qui
  ne pouvait jamais satisfaire le seuil de 50 000 — le throttle
  bloquait presque toutes les soumissions. Corrigé en utilisant
  `global_nodes` (monotone). Débit doublé (~31-36/s → ~60-63/s).
- **Retrait du throttle nœuds** : mesuré sans risque (aucune
  régression NPS mesurable, la queue droppe silencieusement en cas de
  surcharge) — débit porté à ~90-95 tâches/s × 5 candidats.
- **Batch GPU élargi** (`gpu_queue.cpp`) : draine jusqu'à 8 tâches
  (40 positions) avant un seul appel `infer_batch`, au lieu d'un appel
  par tâche de 5 positions — amortit le coût fixe de lancement Metal.
- **Déduplication avant calcul** : `GpuQueue::run()` vérifie la
  fraîcheur dans la GPU TT (et dans le batch en cours) avant
  d'encoder — le taux de recalcul redondant (positions déjà fraîches
  recalculées pour rien) est passé de **~60-70% à 0%**.

## 3. Stratégies de sélection des positions à envoyer au GPU

| stratégie | volume (stores/10s) | `useful_hits` ratio | collision rate | verdict |
|---|---|---|---|---|
| PV uniquement (design initial) | ~700-900 | ~50-160% | ~0% | OK mais peu de volume |
| **near-alpha** (leaf qsearch, score proche d'alpha) | ~24 200 (×27-35) | **3.76%** | **4.5%** | **rejeté** — la plupart des leaves near-alpha sont des branches réfutées, jamais revisitées |
| **transposition prouvée** (TT hit, depth≥4) | ~250-350 | **88-280%** | ~0% | **adopté** — une transposition est prouvée récurrente, contrairement à un near-alpha leaf |

Le hook near-alpha soumettait des positions "fragiles" : alpha-bêta
les abandonne dès qu'elles échouent bas, elles ne sont quasiment
jamais revisitées. Le hook transposition soumet des positions déjà
prouvées récurrentes (transposition = même position atteinte par un
autre ordre de coups), avec une bien meilleure réutilisation.

### Quiet-ification déléguée au thread GPU

Ni les leaves PV ni les transpositions ne garantissaient une position
calme avant. Plutôt que de filtrer côté recherche (coût sur le chemin
chaud), le thread GPU (majoritairement idle, voir section 1) fait sa
propre petite recherche de quiescence (`quietify()`/`quietify_qsearch()`
dans `gpu_queue.cpp`) avant d'encoder — MVV-LVA + SEE, sans TT
partagée, indépendante de `SearchWorker::qsearch`.

**Bug trouvé pendant l'implémentation** : passer le `scratch` du
thread GPU en `VBoard` (nécessaire pour l'éval statique du stand-pat)
comme variable de pile a fait planter le thread (`SIGBUS`). En build
NNUE, `VBoard` embarque les tables de poids **inline** :
`sizeof(NnueEval) = 729 536 octets (~712 Ko)`, `sizeof(VBoard) ≈ 730
Ko` — largement au-dessus des 512 Ko de pile par défaut d'un
`std::thread` sur macOS (vs 8 Mo pour le thread principal). Corrigé en
`static VBoard scratch;`. (Le fichier de poids NNUE sur disque fait
~106 Mo — la majorité des tables sont en fait allouées sur le tas via
des pointeurs, seule une petite partie des couches de sortie est
inline dans l'objet.)

## 4. Design consultatif : ne pas substituer aveuglément le score GPU

Le score GPU (CNN) n'est plus utilisé comme override absolu du score
NNUE de la TT principale. `transp_table.hpp::probe()` classe le score
NNUE et le score GPU chacun contre la fenêtre alpha-bêta courante
(`Low` / `Neutral` / `High`, voir `classify_cut_decision`) :

- **Si les deux classifications s'accordent** → le score GPU est
  utilisé comme raccourci (override), la recherche réelle est
  court-circuitée à ce nœud.
- **Si elles divergent** → le score GPU est ignoré pour ce probe, la
  recherche réelle continue. Ce n'est pas du travail perdu : le
  désaccord détecte une position contestée et déclenche une
  vérification au lieu de faire confiance à un score non vérifié —
  suivi séparément via `disagreement_hits()` (pas confondu avec les
  positions jamais reconsultées, qui elles sont vraiment gaspillées).

## 5. v3 vs v6 : quel modèle CNN utiliser ?

`v6` (design résiduel : `score = nnue_logit + trunk_correction`,
sensé apprendre uniquement l'erreur de NNUE) a été comparé à `v3`
(CNN autonome, entraîné indépendamment, ne voit jamais le score NNUE).

### Corrélation globale avec la vérité terrain (`results_log.csv`)

Même taille d'architecture (14×160) :

| modèle | score_corr (vs Stockfish) | wdl_corr (vs résultat réel) |
|---|---|---|
| v3 (step 300000, final) | **0.7471** | **0.4834** |
| v6 (step 200000, extend-steps final) | 0.7377 | 0.4760 |
| NNUE seule (référence) | 0.6802-0.6833 | 0.4182-0.4232 |

v3 bat v6 sur les deux métriques, malgré l'intuition initiale que
partir de la NNUE (v6) serait un avantage.

### La correction de v6 track-t-elle vraiment l'erreur réelle de NNUE ?

Script : `training/cnn/eval_compare/v6_residual_proportionality.py`.
Résidu réel = `vérité - nnue`, résidu prédit par v6 = `v6 - nnue`.

**En espace cp (brut, sensible à l'échelle) :**
```
Pearson  corr(prédit, réel) = 0.3389
Spearman corr(prédit, réel) = 0.2733
Kendall  tau(prédit, réel)  = 0.1992
same-sign rate: 58.7%  (à peine mieux qu'une pièce jetée en l'air)
v6 réduit |erreur| vs NNUE seul: 47.3% des positions (donc PIRE sur 52.7%)
mean |erreur|: NNUE seul 176.1 → NNUE+v6 165.6 (constant-shift baseline: 176.3)
```
Un correctif constant global (sans aucune info sur la position) ne
fait presque rien (176.3 ≈ 176.1) — donc le gain de v6 n'est pas un
artefact de biais global, il porte un signal réel, mais faible et peu
fiable par position (58.7% de bon sens, corrélations 0.20-0.34).

**En espace WDL (probabilité de victoire, moins sensible à l'échelle
des positions décisives) :** l'écart-type de la correction prédite
s'effondre à **0.0013** (quasi nulle une fois passée dans `tanh`, qui
sature sur les positions déjà tranchées) — `mean |erreur|` NNUE seul
et NNUE+v6 sont **identiques** (0.4038 = 0.4038).

**⚠️ Piège méthodologique rencontré et corrigé en cours de session** :
cette MAE identique semblait contredire `results_log.csv` (v6 wdl_corr
0.4708-0.4760 > NNUE seule 0.4232). Les deux sont vrais simultanément
: `result` (résultat de partie) est un label très bruité (une position
meilleure ne garantit pas de gagner une longue partie) — la MAE contre
ce label est dominée par ce bruit intrinsèque et masque les
différences réelles de qualité entre modèles, alors que la
**corrélation** (déjà la métrique de référence du projet) reste
sensible à ces différences malgré le bruit. **Leçon : ne pas comparer
des modèles via la MAE contre un label WDL bruité — utiliser la
corrélation.**

### Qui prédit le mieux "NNUE se trompe" : v3 ou v6 ?

Script : `training/cnn/eval_compare/v3_vs_v6_error_prediction.py`.
Signal testé : `|nnue - modèle|` (ampleur du désaccord) comme
prédicteur de `|nnue - vérité|` (erreur réelle de NNUE), AUC sur "top
20% pires positions".

| espace | v3 AUC | v6 AUC | gagnant |
|---|---|---|---|
| cp (vs score Stockfish) | **0.8426** | 0.7943 | **v3**, net |
| WDL (vs résultat réel) | 0.3810 | 0.4050 | ni l'un ni l'autre n'est fiable (AUC < 0.5) |

En espace cp — le seul où le signal est réellement mesurable (AUC très
au-dessus du hasard pour les deux) — **v3 prédit mieux les erreurs de
NNUE que v6**, sur toutes les métriques (Pearson/Spearman/Kendall/AUC).
En espace WDL, ni l'un ni l'autre n'est un bon prédicteur (label trop
bruité pour ce diagnostic précis, cf. piège méthodologique ci-dessus).

**Conclusion : v3 reste le meilleur choix**, à la fois comme évaluateur
autonome et comme détecteur de désaccord avec NNUE.

## 6. Raffinement du critère de désaccord (à explorer)

Constat clé (Emeric) : quand NNUE et v3 sont **d'accord pour couper**
(même classification `High` ou `Low`), même si NNUE se trompe en
magnitude, le résultat de la recherche (élaguer cette branche) sera le
même après une exploration plus poussée — une coupe alpha-bêta n'a
besoin que de la bonne *direction*, pas d'un score précis. Chercher
plus loin dans ce cas n'apporte donc probablement rien.

C'est seulement quand les deux modèles sont d'accord pour **ne pas
couper** (`Neutral`/`Neutral`) que la précision du score compte
vraiment : ce score peut remonter tel quel (PV, comparaisons en amont)
plutôt que juste décider d'une coupe.

**Piste retenue à implémenter** : garder le critère de coupe binaire
actuel (accord de classification 3-zones) pour la décision de couper —
mais dans le cas `Neutral`/`Neutral`, appliquer en plus le critère
d'ampleur continue (`|nnue - v3| > seuil`, validé section 5 avec
AUC=0.84) pour décider si le score est fiable ou si une recherche
réelle reste nécessaire. Pas encore implémenté à la fin de cette
session.
