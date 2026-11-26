# Experts par matériel et idées d'autres moteurs à tester

Branche `experts`, ouverte le 26/11/2026. Document de travail : rien n'est implémenté ici.
Principe de la branche : reprendre des idées publiées par d'autres moteurs, les réimplémenter à
notre façon, les entraîner sur nos propres données (Leela) et ne garder que ce qui gagne en parties.
Pas de poids ni d'évaluations de Stockfish (ni copie, ni distillation).

## 1. Pourquoi des experts

L'analyse d'erreur du 25/11 (`reentrainement-long.md`) montre que v7 égale SF v14 sur les positions
équilibrées et que tout l'écart restant est dans les positions déséquilibrées : finales (86 à 91 %
de l'écart), écart de matériel de 1-2 points (82 %), roi exposé, pions passés. Ajouter des
positions déséquilibrées (−0 %), changer la perte (négatif) ou l'échelle (neutre) n'y a rien fait.

Un expert = un jeu de poids distinct pour la partie roi-pièce (HalfKAv2_hm) de la première couche,
choisi par une règle fixe selon le matériel. Un seul expert actif par perspective : le coût par
nœud ne change presque pas, la capacité de la première couche est multipliée. Idée publiée par
Triumviratus (réseaux Consilium et Consilium Plebis, voir section 4).

## 2. Critère recommandé

**Par perspective, selon le matériel hors pions de l'adversaire, avec la présence de la dame
adverse comme première frontière.** C'est la combinaison des deux dernières évolutions de
Triumviratus (8.1 : un expert par camp selon le matériel adverse, +10,3 Elo contre un expert commun
choisi par le nombre total de pièces ; 9.0 : la dame comme frontière).

Pourquoi ce critère plutôt qu'un autre :

- **Par camp plutôt que commun** : chaque perspective choisit son expert selon le matériel qu'elle
  affronte. Les deux perspectives prises ensemble encodent donc le matériel des deux camps, et donc
  le déséquilibre : c'est exactement notre point faible (écart de 1-2 points : 82 % de l'écart avec
  SF v14). Un critère commun (nombre total de pièces) lit pareil « tour contre deux pièces » et
  « deux tours contre deux tours ».
- **Matériel hors pions** : un pion de plus ou de moins ne fait pas changer d'expert (Triumviratus
  le note pour 8.1), ce qui limite les recalculs d'accumulateur à la capture d'une pièce.
- **La dame d'abord** : c'est la frontière qui change le plus le jeu (sécurité du roi, finales).

Découpage de départ proposé (valeurs : cavalier et fou 3, tour 5, dame 9 ; matériel hors pions
d'un camp complet = 31), 6 experts par perspective :

| expert | dame adverse | matériel adverse hors pions |
|---|---|---|
| 0 | oui | ≥ 25 (proche du complet) |
| 1 | oui | < 25 |
| 2 | non | ≥ 13 |
| 3 | non | 6 à 12 |
| 4 | non | 1 à 5 (une pièce : finales de pièce légère ou de tour) |
| 5 | non | 0 (finale de pions côté adverse) |

**Les frontières sont à calibrer sur nos données avant tout entraînement** : compter les positions
de chaque tranche sur les binpacks d'entraînement (outil : `tools/distill/err_buckets.py` donne déjà
le matériel), et ajuster pour qu'aucune tranche ne descende sous ~3 à 5 % des positions. Sinon son
expert manque de données (notre réseau est déjà limité par ses données).

### Garde-fous

- **Initialiser chaque expert comme une copie de v7** : le réseau de départ donne exactement les
  évals de v7, l'affinage spécialise. Pas d'entraînement depuis zéro.
- **Factoriser** : poids = partie commune + correction par expert (comme les buckets de roi de
  nnue-pytorch, coalescés à l'export, coût nul à l'inférence). La partie commune apprend sur toutes
  les positions, ce qui compense la dilution des données.
- **6 experts, pas 20** : plus de tranches = moins de données par expert, plus de recalculs
  d'accumulateur (presque chaque capture de pièce) et ~46 Mo de plus par expert en int16.
  Triumviratus est passé de 8 à 6 par camp entre 8.1 et 9.0.
- **Features de menaces partagées** : seule la partie roi-pièce est dédoublée.

### Plan de test

1. nnue-pytorch : jeu de features `HalfKAv2_hm` + index d'expert (décalage de l'indice par expert,
   facteur commun), script de duplication des poids de v7. Demi-journée.
2. Affinage sur les données BT4 déjà publiées (test80 2024, `xushawn/test80-bt4-relabel`), quelques
   heures de GPU.
3. **Juge avant tout travail moteur** : analyse d'erreur par type de position (scores BT4). On
   attend une baisse nette sur finales et déséquilibres sans dégradation ailleurs. Sinon arrêt
   (~3 à 5 $ dépensés).
4. Seulement si 3 passe : moteur (index d'expert dans la mise à jour, recalcul au changement
   d'expert, tables de rafraîchissement par expert), puis match contre v7.

### Critères écartés

| critère | raison |
|---|---|
| nombre total de pièces (commun aux deux camps) | ne voit pas le déséquilibre ; battu par le critère par camp chez Triumviratus (+10,3) |
| écart de matériel (signé) | change à chaque capture, y compris de pion : recalculs fréquents |
| avancement de la partie (ply) | ne dit rien de la position ; déjà couvert par le matériel |
| structure de pions | change trop souvent, découpage arbitraire |
| 15 à 20 experts | dilution des données, recalculs, mémoire (voir garde-fous) |

## 3. Piste préférée : bloc d'accumulateur conditionnel (26/11)

Idée de l'auteur, discutée le 26/11. Elle corrige le défaut principal des experts : la dilution des
données (chaque expert ne voit qu'une tranche, alors que notre réseau est déjà limité par ses données).

### Principe

- **Les 1024 de v7 restent communs** à toutes les positions : rien n'est découpé.
- **Un petit bloc d'accumulateur en plus**, ~128 neurones appris, alimenté par les **mêmes entrées**
  (roi-pièce et menaces), **actif seulement sous une condition** (déséquilibre de matériel, finale,
  roi exposé).
- Condition vraie : la première couche dense reçoit 512 + 64 entrées (après le produit par paires,
  appliqué au bloc comme aux 1024). Condition fausse : 512 seulement, comme aujourd'hui.

Pourquoi c'est mieux que les experts :

- **Pas de dilution** : les 1024 apprennent toujours sur toutes les positions ; le bloc apprend
  seulement la correction sur les cas difficiles. De la capacité en plus là où l'on perd, rien de
  retiré ailleurs.
- **Spécialisation** : le bloc ne reçoit de gradient que sur les positions où il est actif, donc il
  ne sert pas au cas général (un simple élargissement à 1152 partout n'aurait pas cette propriété).
- **Départ exact depuis v7** : poids du bloc vers la couche suivante à zéro.
- **Plusieurs blocs possibles** (finales, déséquilibre, roi exposé), 64 à 128 neurones chacun, chacun
  avec sa condition.

### Partie non linéaire : trois variantes

| variante | principe | pour | contre |
|---|---|---|---|
| **A (à faire d'abord)** | une seule tête ; les 64 entrées du bloc valent 0 quand il est inactif | gratuit à l'exécution (la couche dense saute les entrées nulles, `SparseL1`) ; la lecture des 512 entrées principales reste commune, le bloc corrige (biais de résidu) ; v7 exact au départ | la tête commune doit servir les deux régimes |
| B | piles de couches distinctes selon la condition (8 × 2 = 16), les piles « actif » prennent 512 + 64 entrées ; « expert non linéaire » | la partie non linéaire se spécialise aussi ; têtes minuscules, coût nul | chaque pile ne voit que ses positions, réapprend à lire les 512 entrées |
| C | tête de v7 + tête de correction (512 + 64 → sortie initialisée à 0) quand la condition est vraie | résidu le plus propre : la tête de v7 n'est jamais modifiée par les cas difficiles | une tête de plus à évaluer, le plus long à écrire (trainer et moteur) |

Ordre : A ; B seulement si A montre un gain réel mais limité ; C seulement si B dégrade les
positions normales.

### Côté moteur

Alcyon tient déjà l'accumulateur PSQT **à part et de façon paresseuse** (`NnueEval::materialize_psqt`).
Le bloc suivrait la même mécanique : mis à jour seulement quand la condition est vraie, en rattrapant
les coups manqués. À 128 de large, une mise à jour coûte ~8 fois moins que celle des 1024 ; si la
condition reste vraie sur une ligne, ~10 % de surcoût d'accumulateur dans ces positions.

### Points à soigner

- **Condition simple et stable**, fondée sur le matériel : elle ne change qu'à la capture d'une
  pièce. Sinon l'éval saute d'un coup à l'autre et le rattrapage se complique.
- **Condition sélective** : sur nos données, ~36 % des positions ont 1-2 points d'écart de matériel
  et ~32 % sont des finales ; à elles deux plus de la moitié. Seuils à serrer et à calibrer sur
  l'analyse d'erreur pour que le bloc reste une spécialisation.

### Test

1. nnue-pytorch : second accumulateur de 128, masque calculé dans le chargeur selon la condition,
   poids nuls au départ (~1 jour).
2. Affinage depuis v7 sur les données BT4, quelques heures de GPU.
3. Juge : analyse d'erreur par type de position ; baisse là où le bloc est actif, rien ne bouge
   ailleurs. Sinon arrêt.
4. Ensuite seulement : accumulateur paresseux dans le moteur, match contre v7.

### Écarté : entrées calculées à la main

Variante discutée puis écartée par l'auteur : greffer ~100 entrées calculées par du code (attaquants
de la zone du roi, règle du carré, déséquilibre par type de pièce...). L'accumulateur apprend
lui-même un signal plus riche ; mieux vaut lui donner de la capacité conditionnelle que des
features faites main.

## 4. Idées d'autres moteurs à tester

Sources : README de Triumviratus (github.com/Tors3/Triumviratus, GPLv3) et code de GaiaChess (github.com/jromang/gaiachess, GPLv3), lus le 26/11/2026.
Les chiffres sont ceux qu'ils publient, pas des mesures à nous. Statut : tout est « à tester ».

### Réseau

| idée | origine | ce que c'est | coût pour nous |
|---|---|---|---|
| experts par matériel, par camp | Triumviratus (Consilium Plebis) | section 2 | moyen |
| poids roi-pièce en int8 | Triumviratus | int8 au lieu d'int16 pour la partie roi-pièce : 2 fois moins de mémoire, permet plus d'experts | moyen (quantification à revoir) |
| sous-réseau pions-roi | Triumviratus | petit bloc (128 unités chez eux) qui lit pions et rois, mis en cache par thread, ajouté à chaque pile de couches ; dans Lex Hortensia, actif seulement en finale de pions | élevé |
| bloc d'entrée `PassedPawns` | Triumviratus (original) | features dédiées aux pions passés ; nos pions passés portent 65 % de l'écart avec SF v14 | moyen |
| bloc d'entrée `PawnPair` | Pawnocchio (aussi Stormphrax, Viridithas) | paires de pions sur la même colonne ou des colonnes voisines | moyen |
| piles de couches selon pièces et dames | Triumviratus (32 piles dans Lex Hortensia), recette SF (`piececount:queen:...`) | choisir la pile de couches aussi selon la présence des dames, pas seulement le nombre de pièces | faible côté moteur, réentraînement |
| features plus récentes de SF | recette SF threats.yaml | `PP_3Wide+K32H3` à la place de `HalfKAv2_hm` | élevé (depuis zéro) |
| buckets de roi × dame adverse (K32Q2) | GaiaChess | 32 cases de roi × présence de la dame adverse : le plus petit « expert » possible (2 jeux de poids) | moyen |
| toute la première couche en int8 | GaiaChess | roi-pièce, paires de pions et menaces en int8 | moyen |
| relations pion-pion retirées des menaces | GaiaChess | elles passent dans le bloc paires de pions (fenêtre de 3 colonnes, 4 560 entrées) | moyen |
| couche à porte | GaiaChess | 64 → porte g et valeur v, h = g²·v, puis sortie avec connexion directe | faible côté moteur, réentraînement |

### Données

| idée | origine | statut |
|---|---|---|
| scores réévalués par Leela BT4 | recette SF | testé : −2 % d'erreur, +9,1 ± 15,7 contre A |
| positions d'arbres de recherche réévaluées par BT4 | recette SF (`xushawn/sf-searchtree-bt4`) | à tester (petit échantillon) |
| réévaluer un corpus avec sa propre recherche | GaiaChess (`relabel.rs`) | à tester : positions des arbres de recherche d'Alcyon, scorées par une recherche d'Alcyon plus profonde (sans SF). GaiaChess : données Leela +26,7 Elo contre 6 milliards de positions maison à recette égale |

### Recherche et gestion du temps

| idée | origine | ce que c'est |
|---|---|---|
| corrections par expert (`CorrPhase`) | Triumviratus | tables de correction séparées selon l'expert actif |
| marges de recherche par expert | Triumviratus | marges RFP, futilité... différentes selon l'expert |
| ordonnancement avec connaissances d'échecs | Triumviratus | bonus pour avants-postes de cavalier, capture d'un fou de la paire |
| réduction « causale » des coups calmes | Triumviratus | à lire dans leur code avant de décrire |
| règle de surprise | Triumviratus | plus de temps quand l'adversaire ne joue pas la réponse attendue |
| moins de temps en retard à la pendule, règles de défense | Triumviratus | modulations de la gestion du temps |
| facteurs multiplicatifs de gestion du temps | Alexandria (cité par Triumviratus) | à lire |
| cache de coups par nombre de nœuds, historique de captures en qsearch | Caissa (cité par Triumviratus) | à lire |
| prefetch de la TT, termes d'ordonnancement | Reckless (cité par Triumviratus) | à lire |
| LMR factorisée | GaiaChess (`lmr_factorized.rs`) | contexte de 9 booléens ; un poids par sous-ensemble de 1 à 3 conditions (91 termes), réglables par SPSA : un contexte rare apprend des contextes courants |

Les deux premières lignes n'ont de sens qu'avec des experts.

## 5. Références

- Triumviratus : README et `source/consilium/` (inférence des experts). Les documents de réseau
  (`docs/networks/NETWORKS.md`, `docs/project/NOVELTIES.md`) et les recettes
  (`Tors3/Triumviratus-Networks`) ne sont pas encore lus.
- GaiaChess : README et `src/` (github.com/jromang/gaiachess, v4.4.4, GPLv3), lus le 26/11/2026.
- Recette SF : `docs/recette-sf.md`.
- Nos mesures : `docs/reentrainement-long.md`.

## 6. Résultats : bloc conditionnel, phase 1 (26/11)

Code : `tools/condnet/` (modèle `cond_model.py` à copier dans nnue-pytorch 4289208 avec
`patch_np.py`). Serveur RTX 5090.

Configuration : v7 figé (89,1 M paramètres), bloc de 128 neurones par perspective (11,1 M
paramètres entraînés : accumulateur du bloc et `l1_extra`, ses poids vers la première couche
dense, paramètre séparé initialisé à zéro), condition « typ » (les deux camps n'ont pas les mêmes
pièces). Données : test80 2024 mars à juin réévalués par BT4 (`-histfix`), filtrés sur la
condition (18,6 % des positions, 3,2 milliards), 12 époques de 100 M, lr 8e-4 × 0,742 par époque,
lambda 1,0. **116 it/s** (contre 14-16 pour un entraînement complet : le gel supprime la
rétropropagation dans l'accumulateur principal) : 11 minutes de GPU.

Deux pièges rencontrés :

- **Bloc mort avec une initialisation aléatoire** : activations trop faibles, nulles après
  quantification et produit par paires, donc aucun gradient. Corrigé en initialisant le bloc comme
  une copie de 64 paires de neurones de v7.
- **Gel et ranger21** : un paramètre figé laissé dans l'optimiseur dériverait par la décroissance
  des poids. Les paramètres figés sont retirés des groupes (`patch_np.py`) ; contrôle : écart nul
  sur les poids de v7 après entraînement.

Vérifications de départ : comptage des pièces depuis les indices = `piece_count` du chargeur ;
condition vraie sur ~21 % des positions de validation (12,8 % des positions calmes) ; modèle de
départ = v7 à 0,0 cp près.

Analyse d'erreur (validation de février BT4, 973 269 positions calmes, même échelle pour les deux
réseaux) :

| | part | v7 | bloc | écart |
|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,46 | −0,4 % |
| condition fausse | 87,2 % | 41,87 | 41,87 | **0,0 %** (gel vérifié) |
| **condition vraie** | 12,8 % | 118,19 | 116,66 | **−1,3 %** |
| écart de matériel ≥ 3 | 2,3 % | 158,31 | 156,92 | −0,9 % |
| finales | 30,4 % | 64,39 | 64,09 | −0,5 % |

Val loss sur les positions sous condition : 0,00509 (v7) → ~0,00502 (moyenne des 8 dernières
époques, bruitée de ±0,00004).

Lecture : gain réel mais petit, environ un dixième de l'écart avec SF v14 sur ces positions
(10 à 13 % d'erreur en moins chez SF v14). Hypothèse : en phase 1 le bloc ne fait que décaler les
entrées des 32 neurones de la première couche dense ; toute la partie non linéaire reste celle de
v7. Les essais coûtent ~10 minutes : variantes à tester ensuite, du plus probable au moins
probable : B (piles de couches propres aux positions sous condition, copies entraînables de v7),
bloc plus large (256, 512), dégel des seules couches denses de v7.

## 7. Résultats : variante B (26/11)

Même bloc, même condition, mêmes données ; en plus, pour les positions sous condition, des **piles
de couches propres** (8 copies entraînables des piles de v7, `cond_stacks`), la sortie du bloc
branchée sur leur première couche. v7 (accumulateur et piles) toujours figé. 11,4 M paramètres
entraînés, lr 5e-4 × 0,774 par époque sur 12 époques, ~100 it/s, ~12 minutes de GPU
(`tools/condnet/run_condB.sh`, `make_cond.py ... typ B`).

| | part | v7 | A | B |
|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,46 | 51,34 (−0,6 %) |
| condition fausse | 87,2 % | 41,87 | 41,87 | 41,87 (0,0 %) |
| **condition vraie** | 12,8 % | 118,19 | 116,66 (−1,3 %) | **115,80 (−2,0 %)** |
| milieu allégé | 20,2 % | 77,70 | 77,35 | 76,98 (−0,9 %) |
| eval 150-400 | 16,5 % | 68,90 | 69,57 | 69,78 (+1,3 %) |

B fait un peu mieux que A (−2,0 % contre −1,3 % sous condition), mais reste loin des 10 à 13 % de
SF v14 sur ces positions, et dégrade encore les avantages nets (eval 150-400).

Loss d'entraînement moyenne par époque **plate dès la première époque** pour A comme pour B
(~4,6e-3, oscillations selon les blocs de données) : le bloc atteint tout de suite ce qu'il peut
apprendre. Pour comparaison, l'affinage BT4 du réseau entier (100 M paramètres libres, 25/11)
n'avait gagné que 1 à 2 % sur ces mêmes positions. Hypothèse : l'erreur restante sur ces positions
n'est pas une question de capacité mais de données (prévisibilité des scores, volume
d'entraînement de SF ~1 000 fois le nôtre). Test qui tranche : une tête plus large (64-128
neurones) sur les positions sous condition ; si la loss d'entraînement reste plate, la limite est
dans les données.

## 8. Résultats : tête large (W) et utilisation du bloc (26/11)

**W** = B avec des piles propres plus larges : première couche 128 neurones (au lieu de 32),
deuxième 64 (au lieu de 32), v7 recopié dans les neurones existants (connexion directe conservée sur
les deux derniers), nouveaux neurones à entrées aléatoires et sorties nulles (départ = v7 à 0,0 cp).
12,5 M paramètres entraînés, ~98 it/s, ~13 minutes (`run_condW.sh`, `make_cond.py ... typ W`).

Loss d'entraînement par époque (× 1e-5, mêmes données dans le même ordre, donc comparables
époque par époque) : W − B = +0,2, −0,2, −0,8, −2,1, −2,8, −3,9, −3,3, −4,0, −4,5, −4,7, −5,2, −5,0.
L'écart grandit régulièrement : gain de capacité. Val loss finale : A 503, B 499, W 493.

Analyse d'erreur (validation de février BT4, positions calmes, même échelle) :

| | part | v7 | A | B | W |
|---|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,46 | 51,34 | **51,20 (−0,9 %)** |
| condition fausse | 87,2 % | 41,87 | 41,87 | 41,87 | 41,87 |
| **condition vraie** | 12,8 % | 118,19 | 116,66 | 115,80 | **114,65 (−3,0 %)** |
| écart de matériel ≥ 3 | 2,3 % | 158,31 | 156,92 | 156,81 | 155,28 (−1,9 %) |
| finales | 30,4 % | 64,39 | 64,09 | 63,98 | 63,72 (−1,0 %) |
| eval 150-400 | 16,5 % | 68,90 | 69,57 | 69,78 | 69,63 (+1,1 %) |

**Utilisation du bloc entraîné de B** (`block_usage.py`, 124 680 positions calmes sous condition) :

- erreur v7 118,19 ; B 115,80 ; **B avec le bloc éteint (piles propres gardées) 117,85** : le gain de
  B vient surtout du bloc (2,05 sur 2,39), peu des piles réentraînées seules ;
- 0 neurone mort sur 128 ; activité médiane 16,5 % des positions ;
- **la variance des 128 sorties est très étalée** : 19 composantes pour 50 %, 83 pour 90 %, 119
  pour 99 %. Le bloc utilise presque toute sa dimension ;
- apport des neurones (extinction un par un) : les 10 premiers portent 51 % du gain, 28 neurones
  coûtent chacun plus de 1 % de l'apport total.

Lecture : contrairement à l'hypothèse « tout vient des données », la capacité limite : élargir la
tête donne un gain qui grandit pendant l'entraînement et se généralise (validation), et le bloc de
128 utilise presque toute sa dimension. Prochain test logique : W avec un bloc plus large (256 ou
512). Restent à surveiller : les avantages nets (eval 150-400) se dégradent légèrement dans toutes
les variantes, et le coût en partie d'une tête 4 fois plus large sur ~13 % des positions.

## 9. Résultats : bloc de 512 (W512) (26/11)

W (tête 128-64) avec un bloc de 512 au lieu de 128 (initialisé avec 256 paires de neurones de v7).
46 M paramètres entraînés, ~80 it/s, ~17 minutes (`run_condW5.sh`, `make_cond.py ... 512 typ W`).

Loss par époque (× 1e-5) :

| | 0 | 3 | 6 | 9 | 11 (fin) |
|---|---|---|---|---|---|
| entraînement B | 464,8 | 469,6 | 453,3 | 463,9 | 460,9 |
| entraînement W | 465,0 | 467,5 | 450,0 | 459,2 | 455,9 |
| entraînement W512 | 464,9 | 465,0 | 446,2 | 452,3 | **449,9** |
| validation B | 510,3 | 514,8 | 502,5 | 495,8 | 498,6 |
| validation W | 510,2 | 513,8 | 498,1 | 489,7 | 493,0 |
| validation W512 | 508,3 | 505,9 | 494,1 | 484,8 | **487,3** |

Analyse d'erreur (février BT4, positions calmes, même échelle) :

| | part | v7 | B | W | W512 |
|---|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,34 | 51,20 | **51,07 (−1,1 %)** |
| condition fausse | 87,2 % | 41,87 | 41,87 | 41,87 | 41,87 |
| **condition vraie** | 12,8 % | 118,19 | 115,80 | 114,65 | **113,68 (−3,8 %)** |
| écart de matériel ≥ 3 | 2,3 % | 158,31 | 156,81 | 155,28 | 153,50 (−3,0 %) |
| finales | 30,4 % | 64,39 | 63,98 | 63,72 | 63,57 (−1,3 %) |
| eval 150-400 | 16,5 % | 68,90 | 69,78 | 69,63 | 69,49 (+0,9 %) |

Lecture : la largeur du bloc paie aussi, mais le rendement baisse : bloc ×4 → −0,8 point
(W → W512), contre −1,0 point pour la tête ×4 (B → W). Sous condition, on est à −3,8 %, environ un
tiers de l'écart avec SF v14 sur ces positions. La dégradation des avantages nets diminue avec la
capacité (+1,3 % → +0,9 %).

Coût prévisible en partie : un bloc de 512 = la moitié de la largeur de l'accumulateur principal,
mis à jour sur les positions sous condition (~13 % des positions calmes), plus une tête 4 fois plus
large sur ces positions. À mesurer dans le moteur avant tout match.

Essai suivant lancé : condition élargie **« deseq »** (pièces différentes ou nombre de pions
différent : 41 % des positions de février), W512, données filtrées `filt_deseq.cpp` (300 M positions
par mois, 1,2 milliard, 12 Go) (`run_condD.sh`).

Disque (serveur 120 Go) : les données filtrées « typ » (31 Go) ont été supprimées pour faire de la
place (régénérables en ~13 minutes avec `filt_typ`) ; les fichiers filtrés sont désormais plafonnés.

### Utilisation du bloc de W512 (`block_usage.py`)

0 neurone mort sur 512 ; 64 composantes pour 50 % de la variance, 324 pour 90 % (63 % de la
dimension, contre 65 % pour le bloc de 128), 472 pour 99 %. Apport très réparti (les 10 premiers
neurones : 22 % du gain). Bloc éteint, piles larges gardées : erreur 125,85, **pire que v7**
(118,19) : les piles larges se sont construites autour du bloc et n'ont pas de sens sans lui.

## 10. Résultats : condition élargie « deseq » (26/11)

W512, condition « deseq » = pièces différentes ou nombre de pions différent (41 % des positions de
février, 41 % des positions calmes : 12,8 % « pièces différentes » + 28,2 % « seulement pions
différents »). Données : 1,2 milliard de positions sous condition (300 M par mois). 12 époques,
~77 it/s.

| | part | v7 | W512 (« typ ») | D (« deseq ») |
|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,07 (−1,1 %) | 51,06 (−1,1 %) |
| matériel identique | 59,0 % | 25,55 | 25,55 | 25,55 |
| seulement pions différents | 28,2 % | 75,96 | 75,96 | 74,61 (−1,8 %) |
| pièces différentes | 12,8 % | 118,19 | 113,68 (−3,8 %) | 116,53 (−1,4 %) |
| finales | 30,4 % | 64,39 | 63,57 | 63,28 (−1,7 %) |
| eval 150-400 | 16,5 % | 68,90 | 69,49 (+0,9 %) | 71,41 (+3,6 %) |

Lecture :

- **même gain total** (−1,1 %) que W512 « typ », mais réparti sur 41 % des positions au lieu de
  13 % : le bloc se dilue (les positions « pièces différentes » ne font plus qu'un tiers de ses
  données) ;
- en partie, « typ » est donc **3 fois moins cher** pour le même gain (bloc actif sur 13 % des
  positions au lieu de 41 %) ;
- « deseq » dégrade nettement les avantages nets (+3,6 %) ;
- la loss d'entraînement de D baisse encore régulièrement en fin de run (378,6 → 355,6) : il
  n'était pas convergé (une seule passe sur ses données).

Conclusion : garder la condition « typ ». Pour les écarts de pions seuls, un second bloc distinct
serait plus logique qu'une condition commune. Essai suivant : W512 « typ » entraîné deux fois plus
longtemps (24 époques), pour mesurer ce que rapporte encore l'entraînement.

## 11. Mesure dans le moteur : la condition est vraie sur ~60 % des évaluations en recherche (26/11)

Mesure faite dans une copie temporaire du code (compteur dans `Eval::eval`, non versionné) : part
des évaluations NNUE complètes pendant le bench où la condition « typ » est vraie.

| bench | évaluations NNUE complètes | condition vraie |
|---|---|---|
| profondeur 12 | 1 263 117 | 58,1 % |
| profondeur 14 | 3 393 162 | 60,1 % |

Contre 12,8 % des positions calmes des données d'entraînement : la recherche évalue surtout des
positions au milieu d'un échange (qsearch, captures), où les pièces sont temporairement différentes.

Conséquences pour le bloc conditionnel « typ » :

1. **coût** : bloc et tête large actifs sur ~60 % des évaluations, pas 13 %. Avec un bloc de 512
   (moitié de la largeur de l'accumulateur, menaces comprises) et une première couche dense 4 fois
   plus large, le surcoût global d'évaluation pourrait dépasser 50 %, à comparer au gain de
   précision (−1,1 % d'erreur au total sur les positions calmes) ;
2. **justesse** : le bloc n'a été entraîné que sur des positions calmes (déséquilibre réel) ; en
   recherche, la plupart de ses activations porteraient sur des déséquilibres passagers jamais vus
   à l'entraînement.

Pistes si l'on poursuit : une condition qui ne s'active pas au milieu d'un échange (par exemple
exiger que le déséquilibre existe aussi dans la position avant la dernière capture, ou n'utiliser le
bloc qu'aux nœuds hors qsearch), ou un bloc beaucoup plus petit. Dans tous les cas, mesurer le coût
réel dans le moteur avant tout match.

## 12. Résultats : W512 entraîné deux fois plus longtemps (24 époques) (26/11)

Données « typ » régénérées avec plafond (`filt_typ_cap.cpp`, 600 M positions par mois, 2,4
milliards, 23 Go). W512, 24 époques, lr 5e-4 × 0,885 par époque (fin ~3e-5), ~35 minutes
(`run_condWL.sh`).

Loss (× 1e-5) : entraînement 465,7 → 422,7 (W512 12 époques : 449,9) ; validation 509,7 →
~478-485 à partir de l'époque 12 (W512 12 époques : 487,3). La validation plafonne dans la seconde
moitié pendant que l'entraînement continue de baisser : début de surapprentissage, ou plafond de ce
que la recette tire de ces données.

| | part | v7 | W512 (12 ép.) | WL (W512, 24 ép.) |
|---|---|---|---|---|
| ensemble | 100 % | 51,65 | 51,07 (−1,1 %) | **50,80 (−1,6 %)** |
| condition fausse | 87,2 % | 41,87 | 41,87 | 41,87 |
| **condition vraie** | 12,8 % | 118,19 | 113,68 (−3,8 %) | **111,56 (−5,6 %)** |
| écart de matériel ≥ 3 | 2,3 % | 158,31 | 153,50 | 151,50 (−4,3 %) |
| finales | 30,4 % | 64,39 | 63,57 | 63,10 (−2,0 %) |
| eval 150-400 | 16,5 % | 68,90 | 69,49 (+0,9 %) | 69,40 (+0,7 %) |

Lecture : l'entraînement plus long rapporte autant que le passage de 128 à 512 (−1,8 point sous
condition). Sous condition, −5,6 %, environ la moitié de l'écart avec SF v14 sur ces positions
(10 à 13 %). Pour comparaison, l'affinage BT4 du réseau entier (25/11) faisait −1 à −2 % sur ces
positions et −2,1 % au total, en 2 h 15. Essai suivant lancé : bloc de 128 entraîné 24 époques
(`run_condWS.sh`), pour situer le compromis coût / précision.

## 13. Bloc de 128 sur 24 époques, et nouvelle validation (26/11)

### WS : bloc de 128, tête large, 24 époques (`run_condWS.sh`)

Validation de février, positions fixes : **−4,4 % sous condition** (W, 12 époques : −3,0 % ; WL,
bloc de 512, 24 époques : −5,6 %), −1,3 % au total. La val loss pendant l'entraînement semblait
plate (487-494) : elle est tirée différemment à chaque époque et bruitée de ±5 points. **Ne plus
juger un plafond sur la val loss d'entraînement, seulement sur l'analyse à positions fixes.**

### Validation tirée des mois d'entraînement (`make_val_tail.py`)

Doutes sur février : fichier `-histfix` de février peut-être traité autrement (la fiche du dépôt
annonce « histfix » à partir de mars), réseaux Leela différents d'un mois à l'autre, et un seul bloc
corrélé de 1,6 M positions (quelques milliers de parties). Nouvelle validation : 12 chunks tirés au
hasard dans la partie **jamais lue** des fichiers de mars à juin (au-delà de la part lue par le
filtre plafonné, + 3 % de marge), 3 par mois : 4,77 M positions, 2,92 M positions calmes. Tous les
réseaux réévalués, aucun réentraînement (`run_valtail.sh`).

| | v7 | A | B | W | W512 | WS | **WL** | D |
|---|---|---|---|---|---|---|---|---|
| ensemble | 50,91 | 50,65 | 50,56 | 50,42 | 50,28 | 50,25 | **50,13** | 50,39 |
| condition vraie (13,4 %) | 113,01 | −1,7 % | −2,3 % | −3,3 % | −4,1 % | −4,4 % | **−5,2 %** | −1,5 % |
| condition fausse | 41,29 | 0 | 0 | 0 | 0 | 0 | 0 | −0,8 % |
| écart de matériel ≥ 3 | 143,90 | −1,3 % | −1,7 % | −2,8 % | −3,8 % | −4,4 % | −4,5 % | −1,0 % |
| finales | 63,69 | −0,6 % | −0,7 % | −1,2 % | −1,4 % | −1,8 % | −2,1 % | −1,4 % |
| eval 150-400 | 68,67 | +0,9 % | +1,1 % | +0,9 % | +0,7 % | +0,9 % | +0,9 % | +3,8 % |

**Correction (26/11, 13 h 45)** : la première version de ce tableau inversait les colonnes WS et WL
(`run_valtail.sh` passait les réseaux dans l'ordre « …, WL, WS, … » à `eval_fens.py` mais les noms
dans l'ordre « …, WS, WL, … » à `an2.py`). Repéré grâce à l'évaluation suivante, où WS donne de
nouveau 108,06. Valeurs corrigées ci-dessus.

(A, B, W, W512 : 12 époques ; WS, WL : 24 époques ; D : condition « deseq », 12 époques.)

Lecture :

- l'ordre A < B < W < W512 et l'effet de l'entraînement long se confirment ;
- **WL (bloc de 512) reste devant WS (bloc de 128)** sur les deux validations : −5,2 % contre
  −4,4 % ici, −5,6 % contre −4,4 % sur février. Le bloc large garde son avantage une fois les deux
  entraînés longtemps ;
- le gain total reste modeste (−1,5 % au mieux) et les avantages nets (150-400) restent
  légèrement dégradés partout.

Choix pratique : à trancher selon le coût dans le moteur : WL (bloc de 512) est le plus précis,
WS (bloc de 128) obtient ~80 % de son gain pour un bloc 4 fois moins cher. Prochaine étape nécessaire avant tout match :
mesurer son coût réel dans le moteur, sachant que la condition est vraie sur ~60 % des évaluations
en recherche (section 11).


## 14. WS2 : WS poursuivi sur des données neuves (26/11)

Données : test80 2024 janvier et février (`-histfix`, 5 M premières positions de février sautées :
la validation de février en vient), 2023 octobre et décembre (BT4, non `-histfix` : réévalués
depuis la position seule), filtrées sur la condition « typ » (`filt_typ2.cpp`) : 2,38 milliards de
positions, toutes neuves pour WS. Préparation en parallèle (téléchargement, filtrage, suppression du
brut, un mois par processus). WS2 repart des poids de WS (24 époques) : 24 époques de plus, lr 2e-4
× 0,905 par époque (fin ~2e-5) (`run_condWS2.sh`), ~25 minutes.

| | v7 | WS | WS2 |
|---|---|---|---|
| fin vierge de mars-juin, sous condition | 113,01 | 108,06 (−4,4 %) | 107,77 (**−4,6 %**) |
| fin vierge de mars-juin, ensemble | 50,91 | 50,25 | 50,21 |
| février, sous condition | 118,19 | 112,97 (−4,4 %) | 112,12 (−5,1 %) |
| février, ensemble | 51,65 | 50,98 | 50,87 |

Lecture : sur la validation impartiale (mars-juin, mois que WS2 n'a pas revus), 2,4 milliards de
positions neuves ne rapportent que 0,2 point au bloc de 128. Sur février le gain est plus grand
(0,7 point), mais WS2 s'est entraîné sur le reste de février. **Le bloc de 128 plafonne vers −4,5 %** :
plus de données ne l'aide plus, sa capacité limite (cohérent avec sa dimension saturée). Le bloc de
512 sur 24 époques (WL) reste meilleur (−5,2 % / −5,6 %).

Incident : en arrêtant la boucle séquentielle pour passer en parallèle, `tmux kill-session` a tué
le filtrage de février en cours, puis le fichier brut a été supprimé ; février refait. Leçon : ne
pas tuer une session qui contient un processus utile, vérifier qu'un processus est vivant avant de
supprimer ses fichiers.

## 15. WL2 : WL poursuivi sur les mêmes données neuves (26/11)

Même protocole que WS2 (section 14), à partir des poids de WL (bloc de 512, 24 époques) :
24 époques de plus sur les 2,38 milliards de positions neuves, lr 2e-4 × 0,905 (`run_condWL2.sh`),
~30 minutes, GPU à ~88 %.

Loss d'entraînement : 463,9 → ~429 (WS2 sur les mêmes données : 473,4 → ~455). Validation de
février (bruitée, favorisée car WL2 s'entraîne sur le reste de février) : ~470 (WS2 : ~481).

| | v7 | WL | WL2 |
|---|---|---|---|
| mars-juin, sous condition | 113,01 | 107,18 (−5,2 %) | **106,64 (−5,6 %)** |
| mars-juin, ensemble | 50,91 | 50,13 | **50,05 (−1,7 %)** |
| mars-juin, écart de matériel ≥ 3 | 143,90 | −4,5 % | −5,3 % |
| mars-juin, finales | 63,69 | −2,1 % | −2,5 % |
| février, sous condition | 118,19 | 111,56 (−5,6 %) | 110,56 (−6,5 %) |
| février, ensemble | 51,65 | 50,80 | 50,67 |

Lecture : sur la validation impartiale, les données neuves rapportent 0,4 point au bloc de 512
(contre 0,2 au bloc de 128) : la largeur aide à absorber plus de données, mais le rendement des
données reste faible. La loss d'entraînement baisse beaucoup plus que la validation : une partie
de ce que le réseau apprend sur ces mois (2023, janvier-février 2024) se transfère mal à mars-juin.

**Meilleur réseau à ce jour : WL2** (bloc 512, tête 128-64, condition « typ », 48 époques au total) :
−5,6 % d'erreur sous condition sur mars-juin, −6,5 % sur février, −1,7 % au total, v7 inchangé
ailleurs. Reste l'étape décisive : le coût dans le moteur (bloc actif sur ~60 % des évaluations en
recherche, section 11).

## 16. Dans le moteur : implémentation, coût et SPRT de WS2 (26/11)

**Implémentation** (branche `cond-block`, bc3d54a puis 353dddd) : `ALCYON_COND_BLOCK=B` à la
compilation. Un seul accumulateur combiné de 1024 + B colonnes, dans l'ordre
`[v7 0..511, bloc 0..B/2-1, v7 512..1023, bloc B/2..B-1]` pour que le produit par paires d'Alcyon
(i avec i + L1/2) associe v7 avec v7 et bloc avec bloc. Les piles de v7 ont des poids nuls sur les
colonnes du bloc ; les piles conditionnelles sont lues dans `<réseau>.cond.nnue`, utilisées si la
condition « typ » est vraie (`NnueEval::cond_typ`). B = 0 : moteur standard, bench 16694 inchangé.
Export depuis PyTorch : `tools/condnet/export_cond.py` (format nnue-pytorch, accumulateur combiné).

**Vérifications** (5 000 positions de la validation mars-juin) :

- export exact en flottant (0,0 cp hors condition, < 0,001 cp sous condition) ;
- moteur avec bloc = moteur v7 à 0 cp près sur toutes les positions hors condition ;
- sous condition, moteur vs PyTorch en **quantification simulée** : 5,2 cp d'écart moyen (v7 : 3,4) ;
  vs PyTorch flottant : 18,4 cp (v7 : 6,6). Le modèle à bloc dépend plus de la quantification, mais
  il a été entraîné avec la quantification simulée (défaut de nnue-pytorch) : c'est elle qui compte.
- erreur en quantification simulée (mars-juin, sous condition) : v7 112,18, WS2 106,90 (−4,7 %),
  WL2 105,88 (−5,6 %) : le gain survit à la quantification.

**Coût** (bench 12, Mac M-series) :

| version | nœuds/s | |
|---|---|---|
| v7 | ~1,48 M | |
| accumulateur 1152, piles de v7 seules (`ALCYON_NO_COND=1`) | ~1,39 M | −6 % |
| WS2 complet (tête 128-64 sous condition) | ~0,73 M | **−51 %** |

Serveur x86 : 436 k → 260 k nœuds/s (−40 %). Le coût vient de la tête large (première couche dense
4 fois plus large, deuxième 8 fois plus lourde), active sur ~60 % des évaluations en recherche ;
l'accumulateur élargi ne coûte que 6 %.

**SPRT WS2 contre v7** (10+0.1, UHO, concurrence 32, serveur) : arrêté à 156 parties,
**−65,4 ± 31,5 Elo** (23 V / 52 D / 81 N). La tête large est abandonnée.

Suite : variante B (bloc 128, piles propres de la taille de v7, coût attendu ~6 %) entraînée
longtemps : 48 époques sur les 4,8 milliards de positions sous condition (`run_condBL.sh`).
Piles conditionnelles configurables dans le moteur (`ALCYON_COND_L2/L3`, 32-32 par défaut).

## 17. Un juge qui prédit l'Elo : positions des arbres de recherche d'Alcyon notées par BT4 (26/11)

Constat de départ : sur les parties Leela (validation habituelle), SF v14 ne fait que ~4 % d'erreur
de moins que v7 alors qu'il vaut ~80 Elo de plus, et nos gains de 1 à 5 % d'erreur ne rapportaient
rien en partie. La mesure ne prédisait pas l'Elo.

**Positions** (branche `searchtree-data`, ea8eb10) : `ALCYON_DUMP_DIR` fait écrire par Alcyon la FEN
d'une évaluation NNUE complète sur `ALCYON_DUMP_RATE` (200) pendant la recherche (pas les positions
en échec, ni les évals relues en TT, ni les évals PSQT seules). Auto-jeu fastchess de v7, 200 000
nœuds par coup, ouvertures UHO_4060_v2, 60 parties en parallèle pendant ~6 minutes (68 parties
commencées) : 11,0 M FEN, 10,8 M uniques (1,8 % de doublons), converties en binpack de positions
isolées (`fen2binpack.cpp` : score 0, ply 9999, résultat 0, coup calme arbitraire, comme
`xushawn/sf-searchtree-bt4`). Répartition : 9 % à 25-32 pièces, 38 % à 17-24, 44 % à 9-16, 9 % à ≤ 8 ;
39 % de positions « pièces différentes ». Limite : seulement 68 parties.

**Notes BT4** : outil de réévaluation de xu-shawn (fork lc0, commit 3f83fab, celui des données SF),
réseau BT4-tf13tune (347 Mo), compilé avec CUDA 13.2 pour sm_120. Débit **~9 000 positions/s**, GPU
à 97 % et 400 W, CPU de lc0 à 73 % : limité par le GPU (≈ 19 Gflop par position, ≈ 170 Tflops, près
du maximum fp16 de la 5090) ; plusieurs processus n'y changent rien. 3 M positions tirées au hasard
notées en ~6 minutes ; 1,7 M ont un score < 10 000 cp.

**Moteurs** : v5, v7, WS2 et le réseau SF v14 (repère uniquement, branche locale `sf14-test`,
réseau converti par un script privé non versionné : SF écrit les PSQT de toutes les features en un
bloc LEB128 après les poids roi-pièce, Alcyon attend un bloc par segment ; version 0x7AF32F20 →
0x6A448AFA). Évals par la commande `eval` (`judge_eval.py`), analyse `judge_an.py`.

Sur **toutes** les positions, l'erreur est ~10 fois celle des parties Leela et dominée par la
tactique (pièces en prise que BT4 voit et que l'éval statique ne voit pas : c'est le rôle de la
qsearch). Sur les **positions calmes** (aucune pièce en prise pour les deux camps, |BT4| < 1500 ;
615 673 positions) :

| | erreur | écart | Elo connu |
|---|---|---|---|
| v5 | 147,38 | | |
| v7 | 131,24 | −11,0 % / v5 | +123 / v5 |
| SF v14 | 121,12 | −7,7 % / v7 | ≈ +82 / v7 (+205 / v5) |
| WS2 | 131,29 | 0,0 % / v7 | précision ≈ 0 Elo (SPRT −65 pour −40 % de vitesse) |

**Environ 11 Elo par point d'erreur** de v5 à v7, 10,6 de v7 à SF v14 : le juge est cohérent avec les
écarts d'Elo connus, contrairement à la validation Leela. Et WS2, −4,7 % sur Leela, ne gagne rien ici.

| positions calmes | part | v5 | v7 | SF v14 | WS2 |
|---|---|---|---|---|---|
| pièces identiques | 87 % | 138,20 | 124,00 | 113,86 (−8,2 %) | 124,00 |
| pièces différentes | 13 % | 211,27 | 181,60 | 171,64 (−5,5 %) | 182,02 |
| 17-32 pièces | 51 % | 157,22 | 142,25 | 132,07 (−7,2 %) | 142,20 |
| ≤ 16 pièces | 49 % | 137,28 | 119,93 | 109,88 (−8,4 %) | 120,09 |

Lecture : sur les positions que la recherche rencontre vraiment, l'avance de SF v14 est **surtout
sur les positions ordinaires** (pièces identiques), pas sur les déséquilibres : le bloc
conditionnel visait la mauvaise cible. Désormais, juger tout réseau candidat sur ce juge (positions
calmes) avant tout match.

Suites : plus de parties et un échantillonnage plus rare pour un juge et un jeu d'entraînement plus
variés ; noter par BT4 seulement les positions calmes (≈ 36 %) pour économiser le GPU (~100 M par
3 heures) ; puis affiner v7 avec une part de ces positions.
