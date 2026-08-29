# v5 — CNN correcteur de résidu de la NNUE

Statut : proposition retenue, pas encore implémentée. Fait suite à
`v4-ideas.md`. Remplace une première version de ce document qui
proposait une fusion via MLP conjoint (accumulateur NNUE + trunk CNN)
— abandonnée au profit d'un design bien plus simple, décrit ici.

## Constat de départ

Deux résultats de l'expérience v4 motivent ce changement de direction
plutôt qu'un v5 "encore plus gros CNN autonome" :

1. **Sur les positions quiètes** (pas d'échec, pas de capture
   disponible — la condition d'arrêt de la recherche de quiescence),
   le CNN v4 dépasse déjà la NNUE (MAE calibré 49,4 vs 53,6 cp, taux
   de victoire 56,7%). Sur les positions non-quiètes (tactiques), la
   NNUE reste nettement devant.
2. **La corrélation d'erreur entre les deux modèles sur les positions
   quiètes est haute (0,79)** — leurs erreurs sont largement
   redondantes, pas complémentaires. Sur WAC (tactique), elle est
   basse (~0,40) — du vrai signal complémentaire.

Conclusion : au lieu d'entraîner un second évaluateur autonome et de
le moyenner avec la NNUE après coup, on entraîne le CNN à faire
directement ce que ces deux chiffres suggèrent : **prédire l'erreur
résiduelle de la NNUE**, pas une évaluation indépendante. Le réseau
apprendra naturellement une correction quasi nulle là où la NNUE est
déjà bonne (positions quiètes, résidu faible) et une vraie correction
là où elle diverge (positions tactiques, résidu informatif).

## Architecture — additive, pas fusionnée

```
score_final = score_NNUE (précalculé, figé)  +  correction_CNN(position)
```

Concrètement, ça se branche directement sur l'architecture `ChessCNN`
existante ([model.py](../../training/cnn/model.py)) : la somme
actuelle est déjà `trunk_logits + psqt_logits`. On ajoute un
**troisième terme additif figé** :

```
logit_final = trunk_logits + psqt_logits + nnue_logit
```

où `nnue_logit` est le score de la NNUE pour cette position, converti
en logit et traité comme une **constante non entraînée** (précalculée
une fois par position, pas un paramètre du modèle — aucun gradient ne
la traverse). Le reste de l'architecture, la loss, la boucle
d'entraînement : **rien ne change** par rapport à v1-v4.

### Pourquoi ce design plutôt que la fusion par MLP (première version de ce doc)

La première version proposait de figer l'accumulateur NNUE, l'exposer
(vecteur `l0`, 1024 de large) à un MLP conjoint prenant aussi les
features du trunk CNN, un par bucket. Deux problèmes ont fait
abandonner cette voie :

1. **Coût d'ingénierie** : nécessitait une réplique PyTorch
   différentiable de l'accumulateur NNUE (le binaire C++ quantifié
   utilisé partout ailleurs dans ce projet ne suffit pas pour
   rétropropager le gradient) — un chantier non négligeable pour un
   gain incertain.
2. **Risque de déséquilibre** : `l0` (1024 dims, déjà très prédictif)
   aurait dominé numériquement les features du trunk CNN (192 dims)
   dans un MLP fusionné — risque réel de sous-utilisation du CNN
   (phénomène documenté sous le nom de "modality imbalance" en
   apprentissage multi-branches), sans solution propre sans perte
   d'info (une projection linéaire pour rééquilibrer les dimensions
   aurait perdu de l'information, sans garantie de résoudre le
   problème).

Le design additif ci-dessus règle les deux : `nnue_logit` n'est pas
une couche à entraîner, juste une valeur précalculée par position —
zéro nouvelle brique PyTorch côté NNUE, et le CNN a un seul objectif
isolé (prédire le résidu), sans jamais pouvoir être "dilué" par une
autre branche dans un MLP partagé.

## Ce qui change dans le pipeline d'entraînement

- **Précalcul de `nnue_logit`** par position, une fois, pas à chaque
  step. Deux options d'implémentation :
  - Intégrer l'inférence NNUE directement dans le chargeur C++
    (`plane_batch.cpp`), en réutilisant le code NNUE déjà présent
    dans le moteur — zéro overhead de communication inter-process,
    et le moteur fait déjà des millions d'évals/seconde en
    recherche réelle donc le coût est négligeable comparé au reste
    du pipeline.
  - Alternative plus rapide à mettre en place mais plus lente à
    l'exécution : passer par le binaire UCI existant (déjà mesuré à
    ~50k pos/s), en cache/precompute avant l'entraînement plutôt
    qu'en ligne.
  - Recommandation : la première option (intégration directe dans le
    chargeur C++) si le volume de données à traiter rend le calcul
    en ligne trop lent, sinon la seconde suffit pour un premier
    prototype.
- **Cible d'entraînement inchangée** — toujours le blend lambda
  score-Stockfish/résultat-réel existant. Le modèle apprend `trunk +
  psqt + nnue_logit ≈ target`, donc implicitement `trunk + psqt ≈
  target - nnue_logit` — un résidu, sans qu'on ait besoin de
  reformuler la loss.
- **`nnue_logit` doit être sur la même échelle logit que le reste**
  — nécessite de reprendre la conversion `sigmoid(nnue_cp / scale)`
  avec un `scale` calibré pour la NNUE (voir la calibration
  cross-validée déjà utilisée dans `per_position_analysis.py` — on a
  déjà les outils, pas de nouveau travail de calibration à inventer).

## Ce qui est réutilisé sans changement (déjà validé dans le projet)

- Le trunk v1 (8 blocs × 96 canaux, SE, plans d'attaque décomposés
  par type de pièce + distance au roi) comme point de départ — voir
  la section taille ci-dessous pour la question ouverte.
- Skip PSQT linéaire séparé (celui du CNN, indépendant du PSQT NNUE).
- 8 buckets de phase, alignés sur la granularité NNUE.
- Optimisations d'entraînement déjà en place (bf16, `torch.compile`,
  `cudnn.benchmark`, TF32).

## Taille du trunk : point à valider empiriquement, pas à deviner

Dans ce contexte (le trunk n'a plus qu'un résidu à capturer, une
tâche plus facile qu'être un évaluateur autonome complet), une taille
inférieure à v1 (8×96) pourrait suffire — mais le signal résiduel
identifié comme utile (positions quiètes, nuances spatiales fines
type plans d'attaque décomposés) n'est pas nécessairement "simple".
Décision : garder v1 (8×96) comme point de départ pragmatique, et
tester une réduction (ex: 3-4 blocs) comme première ablation une fois
l'architecture fonctionnelle, plutôt que de présupposer la bonne
taille.

## Risque : une correction peut se tromper de sens sur une position donnée

Le CNN n'est jamais garanti juste position par position — seulement
en moyenne sur la distribution d'entraînement. Si sa correction va
dans le même sens qu'une erreur déjà présente dans `score_NNUE` sur
une position donnée (plutôt que dans le sens inverse, comme prévu),
le résultat final est pire que la NNUE seule sur cette position
précise — un risque de "double peine". Ce n'est pas spécifique à ce
design (tout ensemble/blend a le même risque), mais reste réel et à
traiter sérieusement avant tout déploiement :

1. **Mesurer avant de déployer** — comparer `score_NNUE + correction`
   contre `score_NNUE` seul sur un grand échantillon représentatif
   tenu à l'écart de l'entraînement (même méthodologie que pour
   v3/v4 : MAE calibré, taux de victoire, corrélation d'erreur). Ne
   jamais supposer que la correction aide sans le vérifier
   directement.
2. **Amortir la correction (shrinkage)** — technique standard en
   gradient boosting : multiplier la correction par un facteur < 1
   (ex: 0.5) pour limiter l'ampleur d'un éventuel excès en cas
   d'erreur du correcteur, au prix d'un peu de gain potentiel.
3. **Intégration moteur "consultative", pas absolue** — envisager
   d'utiliser la correction pour influencer l'ordre des coups ou la
   fenêtre d'aspiration plutôt que de remplacer purement le score
   final, pour limiter les dégâts d'une correction ponctuellement
   fausse.

## Déploiement — ce que ça change pour l'usage final

À l'inférence, il faut toujours calculer `score_NNUE` (rapide,
incrémental, déjà fait par le moteur) **et** `correction_CNN` (lent,
cf. les benchmarks MPS — ~100-800 pos/s selon la taille du trunk).
Le design reste donc contraint par la même conclusion que precedemment
établie : la correction CNN n'est praticable qu'en asynchrone/batché
(ex: réévaluation de la PV ou des feuilles de quiescence), pas comme
remplacement de l'éval synchrone à chaque nœud — ce doc ne change pas
ce constat, juste la façon dont le signal CNN est appris.
