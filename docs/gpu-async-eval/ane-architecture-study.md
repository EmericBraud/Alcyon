# Étude : quelle architecture cible pour une tête policy sur l'ANE

Mesures de **débit** d'inférence sur l'Apple Neural Engine (M4, 16 cœurs) pour
choisir l'architecture du modèle qui produira les hints d'ordonnancement
décrits dans [ordering-hints-plan.md](ordering-hints-plan.md).

Toutes les configurations testées portent le tronc résiduel + une tête policy
AZ (73 plans) + la tête value — c'est-à-dire le modèle complet visé, pas un
tronc nu. Poids aléatoires : on mesure le coût d'une forme, pas la qualité
d'un modèle.

## Méthode, et sa limite

Harnais : PyTorch → `coremltools` → `.mlpackage` fp16 en shape fixe →
`MLModel(compute_units=CPU_AND_NE)`.

**La machine était sur batterie, et la variance est sévère.** Un exemple
mesuré : `8×96` en batch 64 a donné 55863 pos/s dans une passe et 34180 dans
une autre — 63 % d'écart sur une configuration identique. Les tableaux
ci-dessous utilisent donc une **médiane de 3 passes entrelacées** (toutes les
configurations mesurées à chaque passe, pour étaler la dérive thermique sur
tout le panel plutôt que de la concentrer sur les dernières lignes).

Conséquence : les écarts inférieurs à ~15 % ne veulent rien dire ici. Seuls
les facteurs sont exploitables. Tout reste à confirmer sur secteur.

## F1 — L'optimum de batch dépend de la taille du tronc

Batch 128, sans SE, tête AZ. Positions/s :

| tronc | b8 | b16 | b32 | b64 | b128 | b256 | b512 |
|---|---|---|---|---|---|---|---|
| 14×160 (actuel) | 12462 | 15013 | 16247 | **16429** | 15120 | — | — |
| 8×96 | 28902 | 39445 | 47585 | 55863 | **60119** | 52818 | 53735 |
| 6×64 | 44676 | 74875 | 92508 | 111752 | **130831** | 119495 | 109900 |

Le gros tronc plafonne vers 32-64 ; les petits continuent de monter jusqu'à
128, puis redescendent. **Batch 128 est l'optimum des troncs légers, 32-64
celui du modèle actuel.**

Ça corrige une conclusion antérieure de cette session : « 32 est l'optimum
ANE » était vrai du modèle 14×160 seulement, pas de l'ANE. Le réglage de
production actuel (`kAneBatch = 32`) reste correct **pour le modèle actuel** —
b64 y mesure 16429 contre 16247, soit dans le bruit — mais devra être relu si
le modèle change.

## F2 — La concurrence n'apporte rien

8×96, batch 64, N threads soumettant en parallèle :

| threads | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| pos/s | 52696 | 54859 | 52639 | 57240 |

+9 % du premier au dernier, sous le seuil de bruit. **L'ANE est une ressource
sérialisée : elle ne recouvre pas les requêtes concurrentes.** Inutile de
multi-threader l'inférence, et inutile de chercher à cacher la latence par du
pipelining logiciel.

## F3 — Le débit ne suit pas les FLOPs

De 833 à 132 MFLOP par position (6,3× moins de calcul), le débit ne gagne que
3×. Il existe un plancher de latence à 8×8 de spatial : à ces tailles, une
part fixe du temps par appel ne dépend pas du volume de calcul. Réduire le
modèle rapporte, mais de moins en moins.

## F4 — Falaise à 64 canaux (le résultat le plus important)

8 blocs, batch 128, sans SE :

| canaux | MFLOP | pos/s | pos/s par MFLOP |
|---|---|---|---|
| 32 | 20 | 157101 | 7733 |
| 48 | 45 | 134871 | 3022 |
| 56 | 60 | 124728 | 2068 |
| 60 | 69 | 122821 | 1779 |
| **64** | 78 | **111480** | **1422** |
| **68** | 88 | **77511** | **878** |
| 72 | 99 | 74050 | 750 |
| 80 | 122 | 62807 | 517 |
| 96 | 174 | 54845 | 315 |
| 128 | 308 | 39600 | 129 |

**De 64 à 68 canaux : +13 % de FLOPs, −30 % de débit.** Tous les autres pas
sont réguliers ; celui-là est une discontinuité. Très probablement un
alignement 64 voies dans le matériel. 64 est le dernier point du chemin
rapide.

Corollaire à ne pas manquer : une largeur « arrondie à la main » comme 68 ou
72 est le pire choix possible — on paie les FLOPs d'un modèle plus large sans
le débit.

## F5 — À FLOPs constants, la forme vaut 45 % — et à 64 canaux la profondeur est presque gratuite

À ~175 MFLOP, batch 128, sans SE :

| tronc | MFLOP | pos/s |
|---|---|---|
| 4×136 | 177 | 49193 |
| **6×112** | 178 | **65653** |
| 8×96 | 174 | 57905 |
| 12×80 | 181 | 50830 |
| 16×68 | 174 | 45043 |

Lu seul, ce tableau dit « pas trop profond ». Mais en restant sur le chemin
rapide de F4 :

| tronc | MFLOP | params tronc | pos/s (b128) |
|---|---|---|---|
| 6×112 | 178 | 1,35 M | 61340 |
| 8×96 | 174 | 1,33 M | 53140 |
| 8×64 | 78 | 0,59 M | **99370** |
| 12×64 | 116 | 0,88 M | 81657 |
| 16×64 | 154 | 1,18 M | 71213 |
| 20×64 | 192 | 1,47 M | 64697 |

**20×64 a les mêmes FLOPs et le même débit que 6×112, avec 20 blocs au lieu
de 6 et plus de paramètres.** Comparer `16×68 → 45043` (tableau F5) à
`16×64 → 71213` : 13 % de FLOPs en moins pour 58 % de débit en plus. À 64
canaux, narrow-and-deep domine wide-and-shallow sur les deux axes.

## F6 — SE coûte 15-19 %, pas 35 %

Batch 128 :

| tronc | sans SE | avec SE | coût |
|---|---|---|---|
| 8×96 | 57410 | 48625 | −15 % |
| 12×80 | 48294 | 40600 | −16 % |

Une mesure antérieure à batch 32 donnait −35 %. Les deux sont justes : la
réduction globale de SE (pooling sur les 64 cases → 2 FC minuscules →
rediffusion) est un coût de **latence** à peu près fixe par bloc, donc il
s'amortit quand le batch grossit. C'est aussi pourquoi SE est cher alors qu'il
ne pèse que ~0,09 % des FLOPs : il sérialise le pipeline au milieu de chaque
bloc.

## F7 — La taille de la sortie coûte peu

Tronc 6×96 sans SE, batch 32, têtes différentes :

| tête | sorties/pos | pos/s |
|---|---|---|
| value seule | 4 | 65343 |
| from/to factorisée | 128 | 63206 |
| **AZ 8×8×73** | 4672 | 56553 |
| from-to complet | 4096 | 54680 |

La tête AZ coûte 13 % de débit contre value-seule, la factorisée 3 %. Les
FLOPs des deux têtes sont sous 0,5 % du modèle : ce qu'on paie, c'est la
relecture (4672 flottants × batch).

**13 % n'achètent pas la limite de rang 1 de la version factorisée.**
`from[x] + to[y]` ne peut pas exprimer une préférence de destination qui
dépend de la pièce qui bouge, et c'est irrécupérable sans réentraîner. La
tête AZ, indexée spatialement par la case de départ, tient cette information
gratuitement : le « quelle pièce bouge » vient des features du tronc à cette
case.

## Architecture recommandée

**Tronc : 16 à 20 blocs résiduels × 64 canaux, sans SE.**
Conv 3×3, BN repliée dans la conv à l'export (comme aujourd'hui).
~71000 pos/s à 16 blocs, ~65000 à 20. 1,2 à 1,5 M paramètres de tronc.

**Tête policy : AZ 8×8×73** — conv 1×1 de 64 vers 73 plans, indexée par la
case de départ (56 directions type-dame = 8 × 7 distances, 8 cavalier, 9
promotions). **Logits bruts : pas de softmax, pas de masquage dans le
modèle.** Pour de l'ordonnancement seul le rang des coups légaux compte ; le
C++ lit `logits[from_sq][plane]` via une table statique coup→(case, plan) et
trie. Même principe que la sélection de bucket, déjà sortie du modèle.

**Tête value conservée** en multi-tâche : gratuite en FLOPs, elle préserve le
canal actuel, et l'entraînement joint régularise.

**Encodeur : les 31 plans existants, inchangés.** Zéro code C++ nouveau, et
la parité de l'encodeur avec sa référence Python est déjà vérifiée octet par
octet.

**Export : shape fixe, fp16, entrée unique, batch 128.** Les
`EnumeratedShapes` font refuser le modèle par l'ANE **en silence** (voir
`coreml_backend.mm`) ; c'est le piège le plus coûteux de cette exploration.

Débit attendu : ~65-71 k positions/s, contre 12724 pour le modèle actuel doté
de la même tête. Et comme une seule évaluation du parent ordonne **tous** ses
coups, on passe de ~3900 parents/s partiellement ordonnés (4 coups sur ~35) à
~70000 nœuds/s entièrement ordonnés.

## Ce que cette étude ne mesure pas

**La force de jeu.** Tout ci-dessus est du débit. Aucune de ces
configurations n'a été entraînée, donc aucune n'a d'accuracy mesurée. Trois
risques concrets à porter :

Un tronc à **64 canaux est étroit** pour les échecs. La falaise F4 est un
artefact matériel, et rien ne dit que la forme la plus rapide soit la bonne
forme pour apprendre. L'expérience Leela suggère que la largeur compte ;
16×64 a ~4,4× moins de paramètres que le 14×160 actuel.

**Retirer SE est un arbitrage, pas une conclusion.** Les 15-19 % de débit
sont réels ; ce que SE apporte en force ne se mesure qu'à l'entraînement, et
Leela y a trouvé de l'Elo. La comparaison honnête est « deux modèles à budget
de **temps** égal », pas à nombre de paramètres égal — c'est-à-dire un 20×64
sans SE contre un 16×64 avec SE.

**La tête policy demande des labels de coups**, pas seulement des scores. Les
binpacks Stockfish portent le meilleur coup par position, donc les données
existent : entropie croisée one-hot, masquée aux coups légaux à
l'entraînement.

## Chemin critique

Il n'y a **aucun code d'entraînement dans ce dépôt** — `training/` n'existe
pas ici, on ne dispose que du blob de poids exporté. Rien de ce document ne
peut être tenté avant d'avoir débloqué ça, et c'est à faire avant de
retoucher à l'architecture.

Et avant même ça : la **phase 0** du plan d'ordonnancement peut invalider
toute la direction avec deux compteurs et aucune modification de la
recherche. Elle passe d'abord.
