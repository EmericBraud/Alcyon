# v5 — architecture hybride NNUE (figée) + CNN (trunk v1)

Statut : proposition retenue, pas encore implémentée. Fait suite à
`v4-ideas.md` — cette fois, au lieu de continuer à scaler un CNN
autonome, on fusionne l'accumulateur NNUE (rapide, incrémental,
déjà entraîné) avec un petit trunk convolutionnel pour capturer
uniquement le signal résiduel que la NNUE ne représente pas.

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
   redondantes, pas complémentaires. Un simple blend/moyenne des deux
   scores n'apporterait donc qu'un gain de variance limité (la
   réduction de variance d'un ensemble s'effondre quand la
   corrélation entre les deux membres approche 1).

Conclusion : au lieu de faire cohabiter deux évaluateurs entraînés
séparément et de les moyenner après coup, on entraîne un seul modèle
qui **apprend explicitement à ne représenter que le résidu** que
l'accumulateur ne capture pas déjà — plus efficace qu'un ensemble
naïf, et ça règle aussi le problème de déploiement (l'accumulateur
reste incrémental et rapide comme aujourd'hui ; seule la petite
branche CNN a un coût de calcul plein, et seulement quand on
l'invoque).

## Architecture proposée

```
                    ┌─────────────────────────┐
FEN / position ───▶ │ Accumulateur NNUE (FIGÉ) │──▶ features accumulateur (1024, l0/l1)
                    └─────────────────────────┘         │
                    ┌─────────────────────────┐         │
                    │   PSQT NNUE (FIGÉ)       │──▶ psqt_score (par bucket)
                    └─────────────────────────┘         │
                                                          ▼
                    ┌─────────────────────────┐   ┌──────────────┐
Plans 8x8 (33) ───▶ │ Trunk v1 (NON figé,      │──▶│ MLP par bucket│──▶ logit
                    │ 8 blocs x 96, SE, attack/│   │ (8 buckets)   │      │
                    │ king planes)             │   └──────────────┘      │
                    └─────────────────────────┘                          ▼
                                                                  + psqt_score (figé)
                                                                          │
                                                                          ▼
                                                              logit final de victoire
```

### Composants

- **Accumulateur NNUE (`HalfKAv2_hm`, figé)** : poids gelés, tels
  qu'entraînés dans `v3.nnue`. Reste utilisable de façon incrémentale
  (une seule addition/soustraction par coup, comme aujourd'hui dans
  le moteur) — c'est ce qui garde la partie "rapide" de la fusion
  réellement rapide.
- **PSQT NNUE (figé)** : le skip linéaire déjà entraîné, additionné
  directement à la sortie finale (même rôle que `PSQTHead` dans le
  CNN autonome, mais réutilisé tel quel plutôt que réentraîné —
  aucune raison de repayer ce coût, cette partie fonctionne déjà).
- **Trunk v1 (non figé)** : 8 blocs résiduels × 96 canaux + SE, plans
  d'attaque décomposés par type de pièce + distance au roi (les
  plans validés utiles dans v3/v4). Continue d'apprendre — c'est lui
  qui doit se spécialiser sur le résidu.
- **MLP final par bucket** : prend en entrée la concaténation des
  features de l'accumulateur et de la sortie poolée du trunk, une
  instance par bucket de phase (8 buckets, alignés sur le
  `bucket_for_piece_count` de la NNUE — mêmes bornes des deux côtés,
  pour rester synchronisé).

### Point de tap sur l'accumulateur : L0, pas L1/L2 — tranché

On expose au MLP final le vecteur `l0` brut (sortie de la "pairwise
square" de l'accumulateur, 1024 de large — ce que la NNUE elle-même
utilise comme entrée de son propre layer-stack), **pas** les
représentations L1/L2 internes de la NNUE. Deux raisons :

1. **L1/L2 sont déjà "l'opinion finale" de la NNUE**, pas un signal
   brut — donner ça au MLP joint risque d'amplifier le problème de
   sous-utilisation du trunk (cf. section risque ci-dessous) : le
   chemin de gradient le plus facile devient "recopier ce que dit
   déjà L1/L2" plutôt qu'apprendre le résidu.
2. **Double-bucketing redondant** : le layer-stack L1/L2 de la NNUE
   est déjà spécifique à un bucket (8 jeux de poids selon la phase).
   L0, lui, est calculé *avant* la séparation par bucket (accumulateur
   partagé) — le MLP final (lui-même par bucket) peut faire tout le
   travail de spécialisation par phase sans hériter d'un découpage
   déjà fait ailleurs.

L0 est aussi simplement plus riche en information brute (1024 de
large, contre 32+32 pour un L1+L2 déjà compressé pour les besoins
propres de la NNUE, pas nécessairement adaptés à notre tâche jointe).

## Ce qui est réutilisé sans changement (déjà validé dans le projet)

- Skip PSQT linéaire séparé, hors du MLP (évite de mélanger un signal
  linéaire propre dans une transformation non-linéaire).
- 8 buckets de phase (au lieu de 4), alignés sur la granularité NNUE.
- Plans d'attaque décomposés par type de pièce + distance au roi
  (plans validés par ablation sur v3 — cf. discussion sur la
  décomposition, gain net à chaque granularité testée).
- Optimisations d'entraînement déjà en place (bf16, `torch.compile`,
  `cudnn.benchmark`, TF32).

## Risque identifié et mitigation prévue

**Risque** : avec un accès direct aux features de l'accumulateur
(déjà très prédictives) dans le MLP final, l'entraînement pourrait
apprendre à quasiment ignorer la branche CNN — chemin de gradient
plus facile via l'accumulateur, sous-utilisation du trunk. C'est
exactement le phénomène qu'on a déjà mesuré pour le skip PSQT dans le
CNN autonome (contribution modeste, trunk dominant) — risque
symétrique ici, mais avec l'accumulateur comme composant dominant au
lieu du trunk.

**Mitigation envisagée** : ajouter une loss auxiliaire qui force la
sortie du trunk seul (avant fusion) à être prédictive indépendamment,
garantissant qu'il apprend un signal utile avant même la fusion,
plutôt que de compter sur le MLP final pour l'y forcer après coup.

## Coût d'ingénierie

Contrairement aux CNN autonomes (v1-v4), entraîner cette architecture
demande une **réplique PyTorch différentiable de l'accumulateur
NNUE** (pas seulement le binaire C++ d'inférence quantifié utilisé
jusqu'ici pour les comparaisons) — chargée avec les poids de
`v3.nnue`, en float, gelée, mais différentiable pour laisser le
gradient passer jusqu'au trunk pendant l'entraînement conjoint.
Probablement à construire à partir des classes du submodule
`nnue-pytorch` déjà vendoré, plutôt que réimplémenter à partir de
zéro.

## Taille du trunk : point à valider empiriquement, pas à deviner

Discussion ouverte : dans ce contexte (le trunk n'a plus qu'un
résidu à capturer, une tâche plus facile qu'être un évaluateur
autonome complet), une taille inférieure à v1 (8×96) pourrait
suffire — mais le signal résiduel identifié comme utile (positions
quiètes, nuances spatiales fines type plans d'attaque décomposés)
n'est pas nécessairement "simple". Décision : garder v1 (8×96) comme
point de départ pragmatique, et tester une réduction (ex: 3-4 blocs)
comme première ablation une fois l'architecture fonctionnelle,
plutôt que de présupposer la bonne taille.
