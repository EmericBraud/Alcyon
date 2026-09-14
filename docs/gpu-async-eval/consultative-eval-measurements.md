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
réelle reste nécessaire.

## 7. Implémentation v1 du raffinement : rejet + qsearch (jamais commitée telle quelle)

Première implémentation : sur `Neutral`/`Neutral` avec écart > 50cp,
rejeter le score GPU et laisser `negamax` retomber dans son
comportement par défaut (`qsearch` classique). Testé sur WAC (60 puis
150 positions supplémentaires, 800ms/coup) contre la version qui fait
toujours confiance :

- **28/210 positions changent de coup selon la version (~13.3%)** —
  effet réel, pas anecdotique.
- Décodage manuel des cas lisibles : **3 victoires pour le rejet, 5-6
  pour la confiance aveugle** — le rejet perd légèrement, alors que la
  théorie (couper direction seulement, précision seulement sur
  Neutral) semblait solide.

**Pourquoi ce résultat contre-intuitif** : rejeter ne fait qu'annuler
le raccourci gratuit et payer le prix plein d'une vraie recherche à cet
endroit — sous un budget de temps fixe (`movetime`), cette dépense est
retirée du reste de l'arbre, sans garantie de retour. Sur WAC (positions
tactiques choisies pour être informatives), ça se justifie; sur des
positions de partie normale (majoritairement calmes), beaucoup moins.
Confirmé aussi par la non-monotonicité connue des moteurs d'échecs :
un ordre de recherche légèrement différent à un nœud peut faire dévier
le résultat ailleurs (historique/killers/TT), sans lien de cause à
effet direct avec "plus de recherche = mieux".

## 8. Implémentation v2 : extension bornée de 3 plis au lieu du rejet

Au lieu d'abandonner et de laisser `qsearch` s'exécuter, en cas de
désaccord `negamax.cpp` relance `negamax<Us>(kDisagreementExtensionPlies=3,
alpha, beta, ply, true)` — une vraie recherche bornée et prévisible,
pas un rejet sec. Retesté sur les mêmes 210 positions WAC :

- **17/210 diffèrent (8.1%, en baisse par rapport à 13.3%)**.
- Décodage manuel : **6 victoires pour l'extension, 1 pour la
  confiance aveugle** — net renversement par rapport à v1.

**Mais le vrai test (match complet, 300 parties GPU vs NoGPU, 10+0.1,
via `py_scripts/gpu_match_tui.py`) a donné un résultat négatif net :**

```
Elo: -23.20 +/- 18.16   LOS: 0.60%   DrawRatio: 64.00%
```

Statistiquement significatif (intervalle [-41.4, -5.0], ne contient pas
0) : la version GPU **perd ~23 Elo**, malgré la nette victoire sur WAC.

**Diagnostic** : WAC est un banc tactique, pas représentatif d'une
partie réelle — les positions y sont choisies précisément pour rendre
le désaccord NNUE/CNN informatif. En partie réelle, la majorité des
positions sont calmes, où le désaccord est un signal plus faible/plus
bruité (cf. section 6's test à marge étroite, déjà mitigé). Le coût de
l'extension (une vraie recherche de 3 plis, sur le thread de
recherche, à chaque désaccord — mesuré à 10-32 fois par recherche de
8s) s'accumule sur toute une partie et dépasse le gain, même s'il paie
sur un échantillon tactique concentré.

## 9. Implémentation v3 (retenue) : déplacer toute la vérification sur le thread GPU

Constat du problème : tout le coût de la v2 (calcul NNUE de comparaison
+ extension de recherche) tombait sur les **threads de recherche**
(CPU, budget partagé avec le reste de l'arbre), alors que le thread
GPU-eval reste largement idle (voir section 1). Le principe retenu :
**tout déplacer sur le thread idle, remettre le côté recherche à une
confiance inconditionnelle** (comme avant toute la complexité
consultative).

Implémentation (`gpu_queue.cpp`) :
- Après `infer_batch()`, pour chaque position : calcul de l'éval NNUE
  sur la position déjà "quiet-ifiée" (`scratch` a déjà un état d'éval à
  jour, coût marginal — comparable à ce que fait déjà `quietify_qsearch`
  en interne).
- **Accord** (écart ≤ `kNeutralAgreementMaxGapCp`) → stocke le score CNN
  tel quel (`record_agreement()`).
- **Désaccord** → résout avec une vraie recherche bornée
  (`resolve_disagreement<Us>`, minimax complet à profondeur
  `kDisagreementExtensionPlies=3`, fenêtre `[min(nnue,cnn)-50,
  max(nnue,cnn)+50]` pour un élagage alpha-bêta réel malgré l'absence de
  fenêtre "réelle" côté appelant) — stocke le score résolu
  (`record_resolved_disagreement()`).
- Côté recherche (`transp_table.hpp`, `negamax.cpp`) : retour à la
  confiance inconditionnelle d'un hit frais dans la GPU TT — plus
  aucun calcul NNUE ni extension sur le chemin chaud. `probe()` a
  perdu son paramètre `board`/template `Us`, redevenu la version
  simple d'avant tout le mécanisme consultatif.

**Risque identifié avant implémentation (à juste titre)** : le thread
GPU ne peut pas paralléliser une recherche de résolution comme il
paralllélise l'inférence par batch — c'est du calcul séquentiel par
position. Mesuré après implémentation : **`resolved_disagreement_rate`
~68-74%** — le thread passe la majorité de son temps à résoudre des
désaccords, pas juste à faire de l'inférence. Le débit de stores reste
comparable à avant (~265-311/8s), et le NPS des threads de recherche
n'a montré aucune régression mesurable (2.75-3.22M avec et sans GPU,
écart dans le bruit habituel ~5-8%) — donc le thread GPU absorbe bien
ce travail sans affecter les threads de recherche, mais il n'est plus
vraiment "idle" pour autant. **Pas encore testé en match complet** à la
fin de cette session — c'est la prochaine étape logique pour vérifier
si ce déplacement de coût corrige effectivement le -23 Elo mesuré en
section 8.

## 10. Résultat du match v3, et l'erreur de protocole qui a faussé toutes les mesures de la section 9

**Match v3 (300 parties, 10+0.1) : -1.72 ± 28.06 Elo à 203 parties**, soit
rigoureusement rien — cohérent avec le 0.00 ± 28.9 observé à mi-parcours.
Arrêté avant la fin : la revue de code menée en parallèle a montré que le
sous-système était, à l'exécution, quasiment inexistant.

**Erreur de protocole.** Toutes les mesures `gpuevalstats` prises en
envoyant `uci / setoption / position / go / gpuevalstats / quit` d'un seul
bloc dans un pipe sont **invalides**. `go` est non-bloquant dans ce moteur :
`quit` est consommé immédiatement, `shared_gpu_queue().stop()` s'exécute, et
le thread GPU sort de sa boucle avant sa première itération. C'est ce qui
donnait `gpu_thread_busy ≈ 48%` en section 9 (« il reste de la marge ») —
un artefact. Protocole correct : `sleep` entre `go` et `gpuevalstats`, puis
entre `gpuevalstats` et `quit`.

### Mesure réelle (3 s, 10 threads, milieu de partie)

```
nodes 13 205 504   nps 4 443 305
stores=178  useful_hits=18  ratio=10.1%
gpu_thread_busy=99.92%
resolved_disagreement_rate=52.8%  resolve_avg_nodes=1920
push: 334 tentatives, 54 droppées (16%)
```

**18 scores GPU consommés sur 13,2 M nœuds** — 0,00014 % de l'arbre. Le
résultat Elo ~0 ne dit pas « la feature est neutre », il dit « la feature
n'existe pas à l'exécution ». Et le thread était saturé à 99,9 % pour
produire ça : ~90 % de son horloge en calcul scalaire CPU (quietify +
NNUE + résolution), ~10 % en inférence. Le device Metal tournait à ~1,4 %
de sa capacité (60 pos/s contre 4300 pos/s mesurées par `gpubench`).

### Quatre causes, toutes corrigées

1. **Mauvaise clé.** Le score était stocké sous le hash de la position
   *après* `quietify()`. Or la recherche sonde le hash du nœud candidat.
   Rien n'était jamais stocké pour la position que quelqu'un demandait.
   Corrigé : clé prise avant `quietify()`, score renégocié en signe selon
   la parité du nombre de plis joués.

2. **Aucun consommateur là où ces positions sont atteintes.** Les enfants
   d'une feuille de PV sont atteints *dans* `qsearch`, qui ne sondait
   jamais la GPU TT. Consommé comme stand-pat de `qsearch` maintenant —
   c'est la même nature de grandeur (la valeur stockée est quiescée),
   calculée hors thread de recherche. Pas de sémantique de borne attachée.

3. **Offset de calibration.** Mesuré sur quatre recherches (pente,
   intercept, corrélation ajoutés à `gpuevalstats`) : le CNN suit NNUE de
   très près — **pente ~1.0, corrélation 0.90-0.94** — mais se tient
   **~165 cp plus optimiste** pour le trait. Ce n'est pas une erreur
   d'échelle, c'est un offset constant. Corrigé par `kCnnToNnueOffsetCp`
   appliqué au store ; `cnn_to_nnue_intercept` passe de -137..-196 cp à
   -73..+60 cp.

4. **La résolution de désaccord était auto-destructrice.** Elle mangeait
   ~90 % du thread et remplaçait, pour >50 % des positions, le score CNN
   par un minimax 3 plis sans TT ni history — une version dégradée de ce
   que la recherche principale fait déjà. Supprimée. Le désaccord reste
   *mesuré* (`record_cnn_vs_nnue`), plus jamais *agi*.

Supprimé au passage : l'override GPU dans `TranspositionTable::probe()`.
Il appliquait le flag de borne de l'entrée TT (`TT_ALPHA` = « la vraie
valeur est ≤ au score stocké ») à un score statique substitué que ce flag
n'a jamais certifié → coupures injustifiées. Et l'entrée qu'il écrasait
contenait déjà un résultat de recherche d'au moins la profondeur demandée,
strictement meilleur qu'une éval statique.

### Après correction (6 s, 10 threads, 4 positions)

| position | `useful_hits` avant | après | ratio | `busy` |
|---|---|---|---|---|
| milieu 1 | 1   | 34  | 7.0 %   | 95.8 % |
| milieu 2 | 12  | 19  | 4.4 %   | 94.5 % |
| finale   | 3   | 356 | 115.6 % | 77.9 % |
| Kiwipete | 8   | 181 | 94.3 %  | 79.8 % |

NPS inchangé dans le bruit (médianes 4.68M sans / 4.84M avec, 5 runs
entrelacés) malgré la sonde ajoutée sur le chemin chaud de `qsearch`.

### Ce qui reste à décider

- **Le thread de préparation reste le goulot** (78-96 % occupé *sans* la
  résolution). Ce n'est plus l'inférence ni les gates de soumission :
  c'est `quietify` + l'éval NNUE + l'encodage des plans. Élargir les
  gates n'augmentera pas le débit tant que ce coût n'est pas réduit.
- **Corrélation 0.91 entre CNN et NNUE** une fois l'offset retiré : le CNN
  explique ~83 % de la variance de l'éval que le moteur calcule déjà
  gratuitement. Le gain accessible vit dans les 17 % restants — il peut
  être positif comme négatif. C'est la borne haute réaliste de tout ce
  mécanisme, et elle est basse.
