# Elagage appris : bilan et point de reprise

Branche `learned-pruning` (poussee sur GitHub, non fusionnee). Journal
detaille, chronologique, avec toutes les erreurs et corrections :
`docs/learned-pruning.md`. Cette page-ci est le resume pour reprendre.

## En une phrase

Un predicteur appris (regression ou MLP sur l'accumulateur NNUE) reduit
bien l'EBF et achete 1 a 3 plies a budget egal, et il choisit ses noeuds 2 a
4 fois mieux que le hasard -- **mais chaque ply ainsi achete vaut 2.5 a 3 fois
moins qu'un vrai ply** : pas rentable en l'etat.

## Ce qui a ete retenu dans `main`

- **Mate distance pruning** (commit 51e170f) : trouve en chassant une
  position ou la recherche explosait (`8/5p2/q6k/2r5/5p2/2N5/1K6/7q w - - 8 68` :
  33M noeuds a profondeur 11, 9 787 apres). SPRT 54 : +2.7 +- 8.5 sur 2 272
  parties, valide.

## Le mecanisme (dans la branche)

Au point de decision d'un noeud non-PV (apres TT et RFP, avant NMP), un
modele predit P(fail-high) = sigmoid(z) ; selon la confiance, on coupe
(rend la borne), on reduit, ou on cherche normalement.

- **Features** : `static_eval - beta`, eval(ply-2), TT (entree trouvee, borne,
  profondeur, score -- lue par `TranspositionTable::peek`), type de noeud,
  allow_null, materiel, coup precedent, 50 coups, profondeur, ply
  (`fit.features`, 37 scalaires -> 75 entrees avec les interactions `m*x`).
- **Modeles** : regression logistique par profondeur
  (`learned_prune_weights.hpp`) ; MLP 16 sur l0 (entree 1024 octets de la
  pile NNUE) + scalaires (`learned_prune_mlp_weights.hpp`), inference creuse
  (~7 % de nps).
- **Actions** : coupe ; reduction de R plys **verifiee** (re-recherche a pleine
  profondeur si le resultat reduit contredit la prediction) ; reduction
  **graduee** R(z) avec seuils separes fail-high / fail-low. Les re-recherches
  LMR sont exclues (la TT y porte le resultat qui vient de surprendre).
- **Parametres UCI** (build SPSA) : `learned_prune_enabled` (0 eteint, 1 actif,
  2 ombre, 3 hasard), `learned_prune_model`, `learned_prune_t_base`,
  `learned_prune_t_depth`, `learned_prune_max_depth`, `learned_prune_reduction`,
  `learned_prune_graded_slope|z0_high|z0_low|rmax`, `learned_prune_skip_research`,
  `learned_prune_random_permille`, `ordering_noise_seed`.
- **Defaut actuel** : `Enabled=1`, `Model=0`, coupe a T = 0.95 + 0.005 d --
  c'est le reglage du SPRT perdu. **A mettre a 0 avant toute fusion.**

## Resultats, dans l'ordre

| etape | mesure | resultat |
|---|---|---|
| 0 | ce qui arrive au point de decision | l'elagage existant a deja pris les noeuds que l'eval tranche |
| 1 | fits hors ligne (86M noeuds) | la TT porte l'essentiel du signal ; scalaires >> eval seule |
| 2 | SPRT 53 : coupe, regression, T = 0.95 + 0.005 d | **-63 Elo** (1470 parties, echec) |
| 3 | mode ombre | la precision tient en situation (97-99.9 %) : le modele predit bien, les coupes coutent |
| MLP | l0 + scalaires, 25 epochs | loss de test -18 % vs regression convergee ; 16 neurones suffisent, 64 surapprend |
| taux egal | regression / MLP / hasard, R=1, 5 000 positions | a taux de declenchement egal, dans le bruit l'un de l'autre (+-1.1 pt) |
| agressif | budget egal, qualite du coup, 3 000 positions | voir ci-dessous |

**Le test qui tranche** (`movequality.py`, budget egal, perte de score espere
x100, reference = coups evalues a profondeur 15) :

| config | profondeur atteinte | ecart de perte vs base | par ply gagne |
|---|---|---|---|
| base | 12.84 | -- | |
| base, budget x2 | 14.82 | **-0.17** | -0.09 |
| base, budget x4 | 16.89 | **-0.36** | -0.09 |
| MLP graduee z0 = 2 | 15.37 | +0.55 | +0.22 |
| regression R=2 verifiee T = 0.90 | 15.31 | +0.63 | +0.25 |
| MLP R=2 verifiee T = 0.90 | 15.27 | +0.61 | +0.25 |
| regression coupe T = 0.98 | 14.64 | +0.45 | +0.25 |
| regression graduee z0 = 3 | 13.83 | +0.34 | +0.34 |
| hasard R=2, 30 % | 13.75 | +0.37 | +0.40 |
| hasard R=2, 10 % | 13.04 | +0.21 | +1.03 |

Controles : identique 0.000 (exact) ; Hash=4 -0.009 [-0.022, +0.004].

Lecture : un vrai ply vaut ~0.09 point. Un ply achete par l'elagage en coute
0.22-0.25 au mieux (hasard : 0.4-1.0). Il faudrait diviser la perte par
iteration par ~3 pour etre rentable.

## Methode d'evaluation (celle qui marche)

1. **Positions neuves**, jamais vues a l'entrainement (`fens_fresh`, tirees
   plus loin dans le binpack test80 que les positions d'entrainement).
2. **Budget egal, symetrique** (`budget_eval.sh`) : base et config cherchent
   avec le meme budget de noeuds par position (`bench nodesfile`), meme
   binaire. Budget = noeuds(off12) x U, U dans [1, 1.8] par hachage.
   `budget_scale` pour comparer a temps egal (MLP : 0.93).
3. **Qualite du coup, pas accord** (`movequality.py`) : chaque coup joue est
   evalue par une recherche de la position suivante ; perte = E(meilleur) -
   E(coup), E logistique. Un coup different mais aussi bon ne coute rien.
4. **Controles a chaque fois** : identique (doit donner 0 exactement),
   Hash=4 (neutre), hasard au meme regime (plancher), base x2 / x4 (la mesure
   doit recompenser la profondeur).
5. **Comparer les modeles a taux de declenchement egal**, pas a seuil egal
   (`fire_rates.sh`) : a seuil egal le MLP declenche 1.5 a 4 fois plus.

## Les pieges rencontres (a ne pas refaire)

1. **Reference qui prolonge la base** : ref14 repasse par les iterations
   1..12 exactement comme off12 -> toute config qui perturbe l'arbre semblait
   perdre 10 points. Corrige par une reference decorrelee (bruit
   d'ordonnancement, `ordering_noise_seed`) puis par la qualite du coup.
2. **Stockfish comme reference** : mesure en partie le style de Stockfish
   (plafond d'accord a 60 %) ; rejete.
3. **Base limitee en noeuds qui perd son iteration** (`isonode.sh`) : avec 98 %
   des noeuds d'off12 elle retombait au niveau d'off11 -> +2 points factices
   pour toutes les configs. Corrige par l'evaluation symetrique.
4. **Seuil egal au lieu de taux egal** entre modeles.
5. **Parite qui lit la meme entree des deux cotes** : n'aurait pas vu un l0
   perime. Verifie a part (`ALCYON_L0_CHECK` : 0 divergence sur ~150k noeuds).
6. **Precision hors ligne prise pour de la force.**
7. **Reduction sans verification** : 1 % de noeuds au hasard suffisait a
   degrader ; toute reduction doit re-chercher en cas de surprise.
8. **Oublier que le build d'experimentation a le mecanisme actif par defaut** :
   les dumps forcent `learned_prune_enabled=0`.

## Outils (`tools/learned_pruning/`)

| fichier | role |
|---|---|
| `run_dump.sh` | dump des noeuds (scalaires + l0), tous les coeurs |
| `fit.py` | fits hors ligne (a, b-tt, b) par profondeur |
| `export.py` | regression -> header C++ ; `check` : parite |
| `train_mlp.py` | MLP (l0 + scalaires), entrainement long, loss de test par epoch |
| `compare_mlp.py` | MLP vs regression deployee, par profondeur |
| `export_mlp.py` | MLP -> header C++ ; `check` : parite |
| `sweep.sh`, `sweep_report.py` | balayage a profondeur fixe (historique ; biaise, voir pieges) |
| `budget_eval.sh`, `budget_report.py` | evaluation symetrique a budget egal |
| `movequality.py` | qualite du coup a budget egal |
| `fire_rates.sh` | taux de declenchement par config |
| `sf_ref.py` | reference Stockfish (rejetee, gardee pour memoire) |
| `*_configs.txt` | jeux de configs de chaque balayage |

Moteur : `bench nodesfile <fichier>` (budget par position), `bench` affiche
score et profondeur atteinte, `prunestats`, `ALCYON_PRUNE_DUMP`,
`ALCYON_PRUNE_STATS`, `ALCYON_L0_CHECK` (build `-DENABLE_SEARCH_EXPERIMENTS=ON`).

## Etat du serveur (au 2026-09-29)

Instance `i-07e88273d6e44cf2e` (c8g.48xlarge, 192 vCPU, spot), volume de
100 Go monte sur `/data` (dans `/etc/fstab`, `nofail`). **A arreter si on ne
s'en sert pas.** Worker OpenBench dans tmux `wk` (`~/launch164.sh`), a
relancer avec la nouvelle URL ngrok. Donnees :

- `/data/dump_l0/tr.bin`, `te.bin` (+ `.l0`) : 17M / 5.6M noeuds avec l0 --
  base d'entrainement du MLP.
- `/data/mlp/pred_*.pt` : MLP 0 / 16 / 32 / 64 entraines.
- `/data/graded_out`, `/data/aggr_out`, `/data/rate_out` : balayages (logs).
- `~/run/*.bin` : dump scalaire de l'etape 1 (21 Go, regenerable).
- `~/fens_big.txt` (positions d'entrainement), `~/fens_fresh.txt` (neuves).

## Pour reprendre

Il faut diviser par ~3 la perte par iteration. Deux pistes, dans l'ordre :

1. **Changer le label.** Le modele apprend P(fail-high), un proxy. Pour une
   reduction verifiee, ce qui compte est P(la recherche a depth - R rend le
   meme resultat que la recherche complete) -- l'erreur non detectee. Label
   obtenu en lancant les deux recherches aux noeuds tires du dump. Puis meme
   pipeline : entrainement, integration, `movequality.py` avec controles
   base x2 / x4.
2. **Moduler l'existant au lieu d'ajouter.** Utiliser la sortie du reseau pour
   ajuster la reduction LMR ou la marge du RFP, deja reglees et deja
   verifiees, plutot qu'un mecanisme de plus.

Critere de succes, avant tout SPRT : a budget egal, **perte par ply gagne
< ~0.09** (ce que vaut un vrai ply), controles valides.

## A part, trouve en chemin

Dans `move_picker.hpp`, `(thread_id << 32)` n'atteint jamais les 11 bits
gardes par `& 0x7FF` : **tous les threads d'aide de la Lazy SMP ont le meme
bruit d'ordonnancement**. Correctif simple (melanger avec `splitmix64`, comme
`ordering_noise_seed`), a tester en SPRT multi-thread.
