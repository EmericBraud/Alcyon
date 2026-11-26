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

## 3. Idées d'autres moteurs à tester

Source principale : README de Triumviratus (github.com/Tors3/Triumviratus, lu le 26/11/2026, GPLv3).
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

### Données

| idée | origine | statut |
|---|---|---|
| scores réévalués par Leela BT4 | recette SF | testé : −2 % d'erreur, +9,1 ± 15,7 contre A |
| positions d'arbres de recherche réévaluées par BT4 | recette SF (`xushawn/sf-searchtree-bt4`) | à tester (petit échantillon) |

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

Les deux premières lignes n'ont de sens qu'avec des experts.

## 4. Références

- Triumviratus : README et `source/consilium/` (inférence des experts). Les documents de réseau
  (`docs/networks/NETWORKS.md`, `docs/project/NOVELTIES.md`) et les recettes
  (`Tors3/Triumviratus-Networks`) ne sont pas encore lus.
- Recette SF : `docs/recette-sf.md`.
- Nos mesures : `docs/reentrainement-long.md`.
