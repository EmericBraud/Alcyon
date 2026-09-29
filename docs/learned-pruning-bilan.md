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
- **Pas d'extension d'echec** (commit 12b8030) : dans les finales de dames,
  chaque echec prolongeait la recherche (seldepth ~2x, 100M+ noeuds) ; le
  plafond ply < 2 x profondeur ne suffisait pas. SPRT 55 : +3.3 +- 4.7 sur
  7 896 parties, applique pour la robustesse.
- **Boucle infinie de l'aspiration** (commit 3d4ec32) : un score "presque
  mat" plus lointain que la profondeur (souvent venu de la TT d'un coup
  precedent) relancait la meme recherche a l'infini ; `go depth` ne rendait
  jamais la main, et en partie tout le temps restant etait brule.
- **Course sur le plateau partage** (commit 37d4d60) : `main_board` est le
  plateau de l'interface ; la branche ponder annulait son coup APRES avoir
  emis bestmove, pendant que "position" reecrivait deja le plateau, et
  "position" / "ucinewgame" ne joignaient pas la recherche. Sous charge :
  coups rejetes, free(): invalid size, 192 blocages sur 345 parties
  rejouees ; tres probablement les 112 coups illegaux vus en tournoi.
  Apres correctif : 345/345 parties rejouees sans incident, et 400 parties
  fastchess sans coup illegal ni perte au temps.

Les deux derniers sont fusionnes dans `main` (e709aca) sans SPRT (bench
identique) ; un SPRT de non-regression reste a faire.

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

## MoE "reduire est sur" (2026-09-29)

Mecanisme en deux temps, `learned_prune_model = 2` : recherche reduite a
depth - 2, puis un MoE decide s'il fait confiance au resultat. Le label est
"le reduit tombe du meme cote de beta que la recherche complete". Le MoE a
une couche l0 -> 16 partagee et une tete (in -> 32 -> 1) par expert. Les
experts `dsp` sont 20 : tranche de profondeur (1, 2, 3, 4-6, 7+) x cote du
reduit (FL/FH) x phase (> 12 pieces). Chaque expert a son propre seuil,
choisi sur une moitie du test et evalue sur l'autre.

| mesure | resultat |
|---|---|
| hors ligne, 30 % d'economie | 0.069 % d'erreurs (regle pfh : 0.24 %) |
| qualite du coup, budget egal, 12 000 positions | toutes les configs >= 0 (moe +0.03 a +0.16, hasard +0.22, base x2 -0.24) |
| 1 s par coup, 8 positions | profondeur base 15.4 -> moe 17.3, nps inchange |
| **parties**, 5+0.05, 4000 parties, moe0 | **-27.1 +- 6.8 Elo** |
| parties, moe_m100 (plus agressif) | -28.0 +- 6.7 |
| parties, mode ombre (reduit lance, jamais cru) | **-23.0 +- 6.9** |
| parties, ombre + historiques isoles (`probe_isolate=1`) | -34.5 +- 6.7 (pire : isoler est retire) |
| parties, MoE reentraine sur donnees de partie (`moe_game_R2_dsp`) | -31.3 +- 6.9 (pas mieux que moe0) |

Lecture : l'essentiel de la perte vient de **lancer** la recherche reduite
(ombre -23), pas de lui faire confiance (-4 +- 10 en plus, pour ~+1.9 ply).
Les mises a jour d'historique et de killers faites pendant le reduit
aident plutot : les isoler coute 11 Elo.

**Biais de contexte** (`context_bias.py`, 345 parties du moteur, 48 000
positions) : le dataset bench n'est pas le contexte de partie.

| | bench | partie |
|---|---|---|
| TT depth >= depth | 10.0 % | 16.5 % |
| \|eval - beta\| median | 107 | 156 |
| erreur du reduit R=2 | 6.7 % | 4.7 % |

En partie, la TT garde les recherches des coups precedents. Le reduit s'y
trompe moins, mais il y apporte aussi moins. Le MoE reentraine sur 345
parties du moteur fait moins d'erreurs sur ce contexte : 0.23 % contre
0.34 % a 30 % d'economie. En parties, il ne gagne rien pour autant (-31 contre
-27, dans le bruit), ce qui confirme que le cout vient du reduit et pas de
la decision.

Parite C++/Python : `export_moe.py check` sur un dump fait avec
**Threads=1** (le tampon l0 du dump est global, et a 10 threads il se
melange : faux ecarts sur ~7 % des noeuds).

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
| `models/mlp_h{0,16,32,64}.pt` | MLP entraines (etat PyTorch + normalisation), avec leurs journaux `.log` ; `mlp_h16.pt` est celui exporte dans le moteur (`export_mlp.py`) |

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
- `/data/mlp/pred_*.pt` : MLP 0 / 16 / 32 / 64 entraines -- copies dans le repo
  (`tools/learned_pruning/models/`), inutile de les garder sur le serveur.
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

Critere de succes, avant tout SPRT : a budget egal, **ecart de perte < 0**
(pas "< 0.09 par ply" : c'etait faux, l'ecart doit etre negatif). Puis un
match en parties, car le banc ne voit pas le cout du mode ombre.

Apres le MoE : le cout est dans le reduit lui-meme. Pistes : ne lancer le
reduit que sur les noeuds ou un predicteur bon marche (etape 1) le juge
utile, ou revenir a la piste 2 (moduler LMR/RFP/NMP).

## A part, trouve en chemin

Dans `move_picker.hpp`, `(thread_id << 32)` n'atteint jamais les 11 bits
gardes par `& 0x7FF` : **tous les threads d'aide de la Lazy SMP ont le meme
bruit d'ordonnancement**. Correctif simple (melanger avec `splitmix64`, comme
`ordering_noise_seed`), a tester en SPRT multi-thread.
