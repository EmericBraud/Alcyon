# Gestion du temps apprise

Branche `tm-signals`. Remplace la table de stabilite du meilleur coup par un
petit MLP qui predit, a la fin de chaque iteration, la probabilite que le
meilleur coup change a l'iteration suivante.

## Point de depart

Budget souple `horloge / 28 + increment / 2`, limite dure 3x (max 40 % de
l'horloge). La seule modulation etait la duree de stabilite du meilleur coup
racine (facteurs 1.50 / 1.35 / 1.05 / 0.95 / 0.75 selon 0..4+ iterations
stables). On ne decide d'arreter qu'en fin d'iteration, des que le temps
ecoule depasse la moitie du budget module (`SoftStartPercent`).

## Etape 1 : signaux ajoutes a la main

Deux signaux classiques, multiplies au facteur de stabilite (auto-jeu,
10+0.1, serveur Graviton, concurrence 30) :

| Variante | Elo | Parties |
|---|---|---|
| part des noeuds sous le meilleur coup (facteur `(150 - 100 x part) / 100`) | -4.2 +- 18.1 | 500 |
| chute du score (`1 + 0.5 x chute / 100`, borne [0.85, 1.5]) | +8.3 +- 24.1 | 250 |
| les deux | +11.1 +- 23.4 | 250 |

Rien de significatif. Les options restent (`tm_node_*`, `tm_score_*`),
eteintes.

## Etape 2 : donnees

`ALCYON_TM_LOG=1` ecrit une ligne par iteration terminee (profondeur, temps,
noeuds, stabilite, part de noeuds du meilleur et du 2e coup, score et score
precedent, coups legaux, seldepth, pieces, coup). `gen_tm_data.py` rejoue
~1 300 parties (500 d'auto-jeu en 10+0.1, 820 contre SF8 en 60+0.2) coup par
coup, une position sur 4 a partir du coup 8, 1 s de recherche par position,
TT conservee d'un coup a l'autre comme en partie. Soit ~1,06 M iterations.

Cible : le coup change-t-il a l'iteration SUIVANTE ? (taux de base 6,9 %).
Une premiere cible, "differe du coup final de la recherche de 1 s", melangeait
le risque de changement avec le nombre d'iterations restant avant la fin des
1 s ; elle est abandonnee. Iterations de profondeur >= 4, separation
train/test par partie (1 sur 5 en test).

| Predicteur | AUC | Logloss |
|---|---|---|
| table de stabilite (taux par duree, 0..4+) | 0.718 | 0.229 |
| regression logistique, 11 entrees | 0.807 | 0.213 |
| MLP 16 | 0.824 | 0.206 |
| **MLP 32x16** (retenu) | **0.825** | **0.206** |
| MLP 64x32 | 0.827 | 0.206 |

Entrees : profondeur, stabilite (plafond 20), part de noeuds du meilleur et du
2e coup, |score|, chute du score et sa valeur absolue (scores bornes a
+-2000), coups legaux, seldepth - profondeur, pieces, log(1 + noeuds). Les
poids logistiques standardises les plus forts : profondeur (-2.3),
seldepth - profondeur (+1.6), log noeuds (+0.8), stabilite (-0.5), part du
meilleur coup (-0.3).

## Etape 3 : integration

`tm_mlp_weights.hpp` (export_tm.py), evalue une fois par iteration (11 -> 32
-> 16 -> 1, ReLU, sigmoide). Sortie verifiee identique a scikit-learn a 1e-6.
Facteur de temps = `(p / 0.069)^gamma`, borne [0.5, 2], a la place de la
table. Options : `tm_mlp`, `tm_mlp_ref`, `tm_mlp_gamma`, `tm_mlp_min`,
`tm_mlp_max`.

## Etape 4 : parties (auto-jeu, contre la table, aucune perte au temps)

| Reglage | Cadence | Elo | Parties |
|---|---|---|---|
| gamma 0.5 | 10+0.1 | +16.0 +- 16.9 | 500 |
| gamma 0.8 | 10+0.1 | +25.1 +- 19.5 | 500 |
| gamma 0.8 (2e serie) | 10+0.1 | +31.4 +- 19.0 | 500 |
| **gamma 0.8, cumul** | 10+0.1 | **+28.2 +- 13.6** | 1000 |
| gamma 1.1 | 10+0.1 | +27.2 +- 18.7 | 500 |
| gamma 0.8 | 40+0.4 | **-51.9 +- 35.0** | 108 (arrete) |

Plateau entre 0.8 et 1.1 en 10+0.1. **Mais le gain s'inverse en 40+0.4** :
-51.9 +- 35.0 sur 108 parties, arrete.

## Diagnostic : le modele dependait de la cadence

Toutes les donnees venaient de recherches de 1 s. Les entrees les plus
fortes, profondeur et log(noeuds), sont des grandeurs absolues : en 40+0.4
(~1,4 s par coup, et plus en debut de partie) le moteur atteint des
profondeurs que le modele n'a presque jamais vues a ce stade, les juge
"tranchees" et joue trop vite.

## Etape 5 : entrees normalisees, cadences melangees

Normalisation en log, pas lineaire : le temps d'une iteration est multiplie
par l'EBF (~2,2) a chaque profondeur, donc profondeur ~ log(temps) / log(EBF).
Une fraction lineaire ecoule / budget ecraserait toutes les premieres
iterations pres de 0.

- profondeur et log(noeuds) remplaces par
  `iters_left = log((budget + 1) / (ecoule + 1)) / log(2.2)` (iterations que le
  budget souple permet encore) et `log_ebf = log(noeuds / noeuds de
  l'iteration precedente)` ;
- donnees regenerees avec de vraies cadences : base log-uniforme entre 5 s et
  60 s, increment base/100, horloge restante entre 15 % et 100 % de la base,
  budget souple calcule comme `uci.hpp`, recherche de 3x ce budget pour
  connaitre la suite. 1,9 M iterations (1,48 M gardees : profondeur >= 4,
  ecoule <= 2x budget).

Verification (`check_invariance.py`, budget souple comme proxy de la
cadence) : taux de la cible 6,4-7,2 % a toutes les cadences ; 9 entrees sur
11 ont des quantiles quasi identiques ; profondeur (mediane 11 -> 14) et
log(noeuds) (8,1 -> 9,8) derivent comme attendu, d'ou leur retrait.
`iters_left` derive en distribution (un budget long a plus d'iterations au
debut) mais la relation P(cible | iters_left) est stable dans la zone de
decision :

| iters_left | <150 ms | 150-300 | 300-600 | 600-1200 | >1200 |
|---|---|---|---|---|---|
| [0, 1) | 0.078 | 0.075 | 0.063 | 0.068 | 0.063 |
| [1, 2) | 0.081 | 0.077 | 0.074 | 0.067 | 0.074 |
| [2, 3) | 0.077 | 0.072 | 0.088 | 0.071 | 0.066 |

`stable` reste un compte absolu (mediane 9 -> 14 dans la zone de decision),
mais les donnees couvrent desormais toute la plage de cadences.

| Predicteur (cadences melangees) | AUC | Logloss |
|---|---|---|
| table de stabilite | 0.723 | 0.222 |
| logistique | 0.808 | 0.208 |
| **MLP 64x32** (retenu) | **0.825** | **0.201** |

Aussi bon que le modele dependant de la cadence. Poids logistiques
standardises : seldepth - profondeur (+2.9), stable (-0.9), part du meilleur
coup (-0.3). Parite C++ / Python verifiee. Reference du facteur : 6,8 %.

## Etape 6 : parties du modele normalise (gamma 0.8)

| Cadence | Elo | Parties |
|---|---|---|
| 10+0.1 | **+29.3 +- 16.4** | 500 |
| 40+0.4 | +4.3 +- 18.3 | 400 |

Reglage de gamma (modele normalise, reference fixe 6,8 %) :

| gamma | 10+0.1 | 40+0.4 |
|---|---|---|
| 0.5 | - | +13.0 +- 16.9 (400) |
| 0.8 | +29.3 +- 16.4 (500) | +4.3 +- 18.3 (400) |
| 1.1 | +15.3 +- 18.7 (500) | +19.1 puis -7.0, ~+6 (800) |
| **1.4** | **+37.7 +- 17.8** (500) | **+19.1 +- 17.2** (400) |
| 1.8 | +12.5 +- 17.4 (500) | -10.2 +- 17.2 (~400) |

Optimum autour de 1.4 aux deux cadences ; **active par defaut** (`TmMlp = 1`,
`TmMlpGamma = 140`, reference fixe 6,8 %).

Temps consomme (`time_usage.py`) : en 10+0.1 le modele depense ~+11 % sur
les 30 premiers coups pour une depense totale quasi egale ; en 40+0.4 il
depense un peu MOINS au total (-2 a -3 %). Biais suspecte : la reference fixe
(6,8 %) est plus haute que le taux reel dans la zone de decision aux cadences
longues (~5,8 % contre ~7,2 %), donc le facteur moyen y est < 1. Correction
prevue : reference = moyenne glissante des p predits par le moteur
(`tm_mlp_ref=0`), facteur ~1 en moyenne a toute cadence. **Non testee** :
serveur arrete avant ce match (option presente, eteinte).

Resultats, modeles, donnees et PGN du serveur : `alcyon-work/server-backup/`
(hors du repo).


## Pourquoi ca marche alors que l'elagage appris a echoue

La cible est exactement la decision a prendre (une iteration de plus change-t-
elle le coup ?), sans erreurs de gravite cachee ni effet de propagation dans
l'arbre : une mauvaise prediction coute du temps, pas un coup faux. Et le
modele n'est appele qu'une fois par iteration, donc son cout est nul.

## Suite possible

- Arreter en cours d'iteration (aujourd'hui seulement en fin d'iteration).
- Reentrainer sur des donnees generees par la version qui utilise le MLP.
- Ajouter l'ecart de score avec le 2e coup (demande une recherche MultiPV).

## Validation : v6.1 contre Stockfish 8

60+0.2, concurrence 32, memes conditions que v6.0 : **+44.1 +- 19.6** sur 428
parties (155 V / 101 D / 172 N), contre +27.2 +- 15.9 pour v6.0. Valide en v6.1.
