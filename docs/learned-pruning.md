# Elagage appris : "ce noeud vaut-il d'etre calcule ?"

Branche `learned-pruning`. Plan de travail et mesures, tenu a jour au fil
des etapes -- y compris si la reponse finale est non.

## L'idee

Un petit reseau, evalue a l'entree des noeuds, predit le resultat de la
recherche du noeud AVANT de la faire. S'il est assez sur de lui, on rend la
borne sans chercher.

- **Pas de reduction** pour l'instant : le reseau decide seulement "on
  cherche" ou "on coupe". Les reductions (tete multiple, "reduire de N")
  viendront apres, si la coupe marche.
- **Tous les noeuds non-PV** de la recherche principale, a toutes les
  profondeurs. Le seuil de confiance depend de la profondeur : **plus on
  est pres des feuilles, plus on accepte une confiance basse**, parce
  qu'une erreur y coute un petit sous-arbre et qu'un noeud pres des feuilles
  est le plus nombreux, donc le plus rentable a couper.

## Formulation

### Le reseau predit une distribution, pas une decision

"Ce noeud fait-il fail-high ?" veut dire "la valeur de recherche V
depasse-t-elle beta ?". Or beta ne vient pas de la position : c'est ce que
le reste de l'arbre a deja trouve, relativement a la racine. La meme
position est a couper sous une fenetre et a chercher sous une autre. Un
reseau qui ne verrait que la position (l'accumulateur) ne peut donc pas
repondre seul -- mais il n'a pas besoin de tout l'historique du chemin
non plus : a profondeur fixe, V ne depend que de la position (sauf
repetitions et regle des 50 coups), et beta resume la comparaison a la
racine.

D'ou la separation : le reseau predit la **distribution de V** a partir
de ce que le noeud sait, et les bornes n'interviennent qu'au moment de
poser la question.

```
V ~ centree sur eval + biais, de largeur sigma
P(V < x) = F( (x - eval - biais) / sigma )    // F = sigmoide

biais, sigma = reseau(accumulateur, depth, scalaires)   // sigma > 0

P(fail-high) = 1 - F(beta)
P(fail-low)  = F(alpha)
P(exact)     = F(beta) - F(alpha)             // seulement en PV
```

- **Le reseau ne voit ni alpha ni beta.** Il repond pour n'importe quelle
  fenetre, nulle ou non.
- **La dependance a `eval - beta` est cablee**, et toujours dans le bon
  sens : plus de marge, plus de fail-high. Le reseau n'a pas a la
  redecouvrir ; il n'apprend que ce que l'eval ignore.
- **Ce que l'accumulateur apporte, c'est `sigma`** : la volatilite de la
  position. Pieces en prise, roi expose, clouages -- l'eval statique est
  la meme, mais une recherche a profondeur d peut la renverser. `sigma`
  petit : position calme, on peut couper. `sigma` grand : tactique, on
  cherche. `biais` capte les erreurs systematiques de l'eval (le trait,
  une menace que l'eval ne voit pas, un score TT deja connu).
- **Le RFP et le razoring actuels en sont le cas particulier** ou `biais`
  et `sigma` ne dependent que de la profondeur. C'est la ligne de base a
  battre (etape 1).

### Alpha, et les noeuds non-PV

Dans un noeud non-PV la fenetre est nulle (`alpha = beta - 1`) : fail-low
(`V <= alpha`) et "pas fail-high" (`V < beta`) sont la meme chose. Alpha
n'apporte rien et une seule probabilite `p = 1 - F(beta)` decide des deux
cotes :

```
si p >= T(depth)       -> return beta       // comme RFP / NMP
si p <= 1 - T(depth)   -> return alpha      // comme razoring
sinon                  -> recherche normale

T(depth) = clamp(T0 + T1 * depth, 0.5, 1)  // T1 > 0 : plus exigeant en haut
```

Le meme mecanisme generalise donc RFP, razoring et NMP. `T0` et `T1` sont
les seuls parametres laisses au SPSA -- deux, pas des milliers (voir "Ce
qu'on ne fait pas"). Alpha ne compte vraiment qu'en PV (trois issues),
exclu pour l'instant ; la formulation le couvre deja le jour ou on y va.

Choix de depart, chacun revisable :

- **Fail-hard** (`beta` / `alpha`), comme le RFP actuel.
- **Rien en TT** pour un noeud coupe par le reseau : une borne devinee ne
  doit pas se propager a d'autres chemins. Le jour ou ca marche, stocker
  avec une profondeur 0 est une experience separee.
- **Exclus** : racine, PV, noeud en echec, fenetres de mat, recherche
  singuliere (`excluded_move`), qsearch (le stand-pat y joue deja ce role).
- **Place** : apres le probe TT et le RFP, AVANT le NMP. Le NMP coute une
  recherche, le reseau quelques dizaines de ns : autant l'economiser. En
  aval du RFP, le reseau ne voit que les noeuds qui ont survecu a
  l'elagage existant -- c'est aussi exactement la distribution qu'on
  logge pour l'entrainer, donc pas de decalage a cet endroit.

## Features d'entree

Aucune entree ne contient alpha ou beta : les grandeurs de score sont
relatives a l'eval statique, pas a la fenetre. Scalaires du noeud (tous
deja disponibles a l'endroit de la decision) :

- `depth`, `ply`, camp au trait
- `eval(ply) - eval(ply-2)` (et si elle est connue : le cache est
  paresseux) -- c'est `improving` en continu
- TT : entree trouvee, type de borne, `tt_depth - depth`,
  `tt_score - static_eval`. Lu par `TranspositionTable::peek`, pas par
  `probe` : au point de decision `probe` a deja echoue, mais une entree
  trop courte ou dont la borne ne coupait pas reste de l'information de
  recherche deja payee, que la position seule ne contient pas
- type de noeud (cut / all) : l'issue que le parent attend
- `allow_null` (le parent vient d'un null move)
- nombre de pions, cavaliers, fous, tours, dames de chaque camp
- coup precedent : piece deplacee, piece capturee
- compteur des 50 coups : un des rares cas ou V depend du chemin

Absents du dump v1, a ajouter si (b) montre un signal : nombre de coups
legaux (demande une generation), repetition possible, le coup precedent
donnait-il echec (toujours faux ici : les noeuds en echec sont exclus).

Plus l'accumulateur NNUE du noeud (L1 = 1024 par camp), qui encode la
position -- tension tactique comprise -- et qui est deja calcule des que
le noeud evalue. A mesurer : combien de noeuds atteignent la decision sans
avoir evalue (cout d'une mise a jour d'accumulateur en plus).

Architecture de depart : accumulateur -> 16 (int8, SIMD) concatene aux
scalaires -> 16 -> 2 sorties (`biais`, `log sigma`). De l'ordre de 17k
MAC, soit quelques dizaines de ns : a comparer au nps (environ 600k en
bench, donc ~1.7 us par noeud). `eval` n'entre pas dans le reseau : elle
est ajoutee a la sortie, dans la formule de F.

## Donnees et entrainement

**Labels.** Mode de dump (build d'experimentation,
`ALCYON_PRUNE_DUMP=<fichier>`, `ALCYON_PRUNE_DUMP_EVERY=N`) : au point de
decision, un noeud est tire avec la probabilite `min(1, 2^(depth-1) / N)`
-- les noeuds pres des feuilles sont des millions, ceux du haut quelques
milliers ; on ecrit les features, l'eval statique et beta,
puis on laisse la recherche se derouler normalement et on ecrit son
resultat (fail-high, fail-low, ou coupe par le NMP -- compte comme
fail-high) et la taille du sous-arbre. Enregistrements de 112 octets
(`search::PruneRecord`, repete dans `tools/learned_pruning/fit.py`) dans
un tampon statique ecrit par `fwrite` -- pas d'allocation a l'execution.
Mono-thread. Le dump ne change pas l'arbre (meme compte de noeuds au
bench), et son nombre d'enregistrements a `EVERY=1` egale exactement les
noeuds que `prunestats` voit passer le RFP.

Reproduire :

```
cmake -B build-exp -DCMAKE_BUILD_TYPE=Release -DENABLE_GUI=OFF -DENABLE_SEARCH_EXPERIMENTS=ON
cmake --build build-exp -j8
# positions : dump_val_fens (training/cnn/data_loader) sur le binpack,
# dedoublonnees, coupees en blocs contigus 75 % train / 25 % test
printf 'bench 12 fens_train.txt\nquit\n' | ALCYON_PRUNE_DUMP=train.bin ALCYON_PRUNE_DUMP_EVERY=3 ./build-exp/alcyon
printf 'bench 12 fens_test.txt\nquit\n'  | ALCYON_PRUNE_DUMP=test.bin  ALCYON_PRUNE_DUMP_EVERY=3 ./build-exp/alcyon
python3 tools/learned_pruning/fit.py train.bin test.bin
```

**Positions.** Les binpacks existants, a une profondeur fixe moderee, ou
les parties d'auto-jeu d'OpenBench.

**Donnees censurees.** Une recherche en fenetre nulle ne donne jamais V,
seulement de quel cote de beta il se trouve. On n'a donc pas de cible de
regression, mais une observation "au-dessus / en dessous" d'un seuil
different a chaque noeud. C'est suffisant : la perte est l'entropie
croisee de `1 - F(beta)` contre le resultat observe, et c'est la variete
des betas d'un noeud a l'autre (a eval egale) qui permet d'apprendre la
forme de la distribution -- `biais` et `sigma` -- et pas seulement un
seuil.

**Entrainement** sur le serveur GPU, a cote de `training/nnue-pytorch`,
avec la perte ci-dessus. Les erreurs n'ont pas toutes le meme prix : on
pondere par la taille du sous-arbre (ce qu'une coupe juste economise, ce
qu'une coupe fausse rate). Reseau fige ensuite, quantifie en int8.

**Decalage de distribution.** Une fois le reseau actif, il coupe aussi des
descendants, donc l'arbre change. Re-dumper avec le reseau actif et
re-entrainer une ou deux fois.

## Etapes

0. **Compter ce qui arrive au point de decision.** Fait, voir ci-dessous.
   C'est la lecon de `docs/lmr-history-modulation.md` : on mesure avant
   d'ecrire le mecanisme.
1. **Dump + fits hors ligne**, tous avec la meme forme `F` et la meme
   perte, du plus simple au plus riche :
   - (a) `biais`, `sigma` fonctions de la profondeur seule : ce que RFP
     et razoring savent deja ;
   - (b) regression logistique sur les scalaires, un fit par tranche de
     profondeur (1, 2, 3, 4-6, 7+) : rapide, convexe, et ses poids disent
     quelles features comptent ;
   - (c) le petit reseau vise, scalaires + accumulateur. Une sonde
     lineaire sur l'accumulateur raterait les interactions (une tension
     qui ne compte qu'a faible marge), d'ou le MLP directement.

   La question precise : **l'accumulateur predit-il `sigma` mieux qu'une
   fonction de la profondeur ?** Critere de poursuite, a chaque
   profondeur : part du sous-arbre coupable avec une precision
   >= T(depth). Si (b) et (c) ne battent pas nettement (a), on s'arrete
   la : le reseau n'aurait rien a dire que l'elagage actuel ne dit deja.
2. **Integration** du reseau (inference int8, accumulateur, features),
   derriere une option UCI eteinte par defaut. Verifier l'impact nps.
3. **Recompense dense hors ligne** pour choisir `T0`, `T1` sans jouer de
   parties : sur un jeu de positions fixes, comparer
   `search(d, reseau actif)` a `search(d, reseau eteint)` --
   meme meilleur coup ? combien de noeuds en moins ?
4. **SPSA sur `T0`, `T1`**, puis **SPRT** contre la version de base, et
   un controle contre Stockfish 8 (pas seulement de l'auto-jeu).
5. Plus tard, hors perimetre : noeuds PV (meme reseau, question posee a
   alpha ET beta), tete de reduction, qsearch, affinage des poids par la
   recompense dense (ES).

## Ce qu'on ne fait pas, et pourquoi

**Pas d'apprentissage par renforcement sur le resultat des parties.** Une
partie represente environ 10^8 decisions d'elagage pour un seul resultat,
et le SPSA a deja besoin de ~30 000 parties pour 4 parametres (tune 43).
Sur des milliers de poids, avec ~23 000 parties par jour, ca ne converge
pas. Le resultat des parties ne sert qu'aux 2 parametres de l'etape 4.

## Etape 0 : mesures

Build `-DENABLE_SEARCH_EXPERIMENTS=ON`, `ALCYON_PRUNE_STATS=1`, commande
UCI `prunestats` apres `bench 16` (16 positions, Threads=1, 8.85M noeuds).
Le nombre de noeuds est identique avec et sans les compteurs (verifie
position par position a `bench 12`) : la mesure ne change pas l'arbre.

Perimetre compte : noeuds non-PV, hors echec, ply > 0, depth >= 1, hors
recherche singuliere, qui ont passe le probe TT (le razoring, place avant,
est compte aussi).

**Comment se terminent les noeuds :**

| depth | n | razor | RFP | NMP | cherches |
|---|---|---|---|---|---|
| 1 | 1 706 675 | 5.5 % | 84.8 % | 0 % | 9.7 % |
| 2 | 992 385 | 5.7 % | 77.9 % | 0 % | 16.3 % |
| 3 | 619 551 | 0 % | 74.1 % | 1.3 % | 24.6 % |
| 4 | 362 164 | 0 % | 70.0 % | 3.1 % | 26.9 % |
| 5 | 237 518 | 0 % | 66.6 % | 4.4 % | 29.0 % |
| 6 | 172 748 | 0 % | 65.6 % | 4.1 % | 30.3 % |
| 7 | 99 654 | 0 % | 0 % | 51.5 % | 48.5 % |
| 8 | 51 341 | 0 % | 0 % | 46.5 % | 53.5 % |
| 9 | 27 101 | 0 % | 0 % | 45.9 % | 54.1 % |
| 10+ | 22 219 | 0 % | 0 % | 37.5 % | 62.5 % |

**Taux de fail-high des noeuds cherches, par `static_eval - beta` :**

| depth | < -400 | -400..-200 | -100..-50 | -50..0 | 0..50 | 100..200 | 200..400 | >= 400 |
|---|---|---|---|---|---|---|---|---|
| 1 | -- | 38 % | 41 % | 54 % | 75 % | -- | -- | -- |
| 2 | -- | 31 % | 40 % | 54 % | 76 % | -- | -- | -- |
| 3 | 13 % | 29 % | 41 % | 55 % | 73 % | 87 % | -- | -- |
| 4 | 11 % | 30 % | 44 % | 57 % | 74 % | 88 % | -- | -- |
| 6 | 14 % | 41 % | 50 % | 58 % | 74 % | 84 % | 91 % | -- |
| 7 | 13 % | 40 % | 49 % | 59 % | 75 % | 84 % | 93 % | 98.8 % |
| 8 | 13 % | 42 % | 53 % | 59 % | 75 % | 86 % | 92 % | 98.8 % |
| 9 | 14 % | 41 % | 49 % | 60 % | 74 % | 81 % | 91 % | 98.2 % |

(colonnes -200..-100 et 50..100 omises, elles s'intercalent. Sortie complete,
avec la part du sous-arbre par bucket : relancer
`prunestats`.)

**Ce que ca dit :**

1. **L'elagage existant a deja pris les noeuds faciles.** Pres des
   feuilles, 84 a 90 % des noeuds sont coupes (RFP surtout) avant toute
   recherche. Ceux qui restent sont, par construction, ceux ou l'eval ne
   tranche pas : aucun bucket ne depasse 91 % ni ne descend sous 11 % a
   depth <= 6. Meme phenomene que pour la LMR-history : l'information
   "eval" est consommee en amont.
2. **Donc le reseau ne peut gagner qu'avec de l'information hors eval.**
   Autour de `eval ~ beta` (-50..+50), les noeuds sont a 55-75 % : c'est la
   que se trouve ~35 % du sous-arbre, et c'est la que l'accumulateur et la
   TT doivent faire la difference. L'etape 1 le mesure avant d'integrer
   quoi que ce soit.
3. **Un trou simple a depth >= 7 :** le RFP s'arrete a depth 6. Au-dessus,
   les noeuds cherches avec `eval - beta >= 400` echouent haut a 98-99 %,
   et le NMP n'en rattrape qu'une partie. C'est 5 a 6.5 % du sous-arbre a
   ces profondeurs. Prolonger le RFP (`MaxDepth` 6 -> 8) le comblerait,
   mais le gain attendu (1 a 3 % des noeuds, quelques Elo au mieux) ne
   vaut pas un SPRT : non retenu. Le reseau le couvrira de toute facon.
4. **Cote fail-low**, le meilleur bucket (`< -400`, depth 3-6) reste a
   11-14 % de fail-high : le razoring actuel n'a pas de marge evidente a
   recuperer avec l'eval seule.

## Etape 1 : resultats (features scalaires)

Dump sur le serveur 192 coeurs (`tools/learned_pruning/run_dump.sh`) :
positions du binpack test80, `bench 16`, `EVERY=1024`. Train : ~34 700
positions, 86M noeuds ; test : ~14 300 autres positions, 35M noeuds.
Fits : `fit.py`, 2M noeuds de train au plus par profondeur.

**Part du sous-arbre coupable a precision >= 99 % / >= 98 %**, noeuds
vraiment cherches seulement (`FIT_EXCLUDE_NMP=1` : les noeuds coupes par
le NMP sont des fail-high faciles et deja bon marche) :

| depth | (a) eval seule | (b-tt) scalaires sans TT | (b) scalaires + TT |
|---|---|---|---|
| 1 | 0 / 0 % | 0 / 0 % | 0 / 16.5 % |
| 2 | 0 / 0 % | 0 / 0.2 % | 0 / 12.0 % |
| 3 | 2.1 / 10.5 % | 11.6 / 20.2 % | 21.1 / 35.7 % |
| 4 | 0 / 8.7 % | 6.6 / 16.6 % | 19.8 / 36.2 % |
| 5 | 0 / 8.0 % | 5.2 / 14.7 % | 20.1 / 39.3 % |
| 6 | 0 / 7.0 % | 4.8 / 14.5 % | 21.7 / 46.0 % |
| 7 | 1.3 / 9.3 % | 9.3 / 19.3 % | 28.4 / 52.9 % |
| 8 | 1.0 / 8.4 % | 8.2 / 18.8 % | 25.0 / 52.6 % |
| 9 | 0.1 / 7.3 % | 8.0 / 19.3 % | 27.1 / 55.6 % |
| 10+ | 0 / 0.6 % | 6.9 / 18.1 % | 21.8 / 53.6 % |

Avec les noeuds NMP inclus (le reseau place avant le NMP economiserait
aussi sa recherche), les chiffres de (b) montent a depth >= 7 (48-58 % a
99 %), ceux de depth <= 2 ne changent pas (pas de NMP a ces profondeurs).

**Ce que ca dit :**

1. **Le critere de poursuite est rempli** : (b) bat nettement (a) a toutes
   les profondeurs. A 99 % de precision, l'eval seule ne coupe presque
   rien ; les scalaires en coupent 20 a 28 % du sous-arbre de depth 3 a 10+.
2. **L'essentiel vient de la TT** (b-tt -> b : x2 a x4). Ce sont des
   entrees que `probe` ignore : trop courtes, ou dont la borne ne coupait
   pas -- typiquement l'iteration precedente, ou la recherche reduite de
   la LMR juste avant sa re-recherche. De l'information de recherche deja
   payee, que le moteur jette aujourd'hui.
3. **La position seule apporte aussi** (a -> b-tt : 5 a 12 % a 99 %), sans
   l'accumulateur.
4. **Depth 1-2 resistent** : rien a 99 %, 12-17 % a 98 %.
5. **Ce que la precision ne dit pas** : 1 % d'erreurs a depth 10 n'a pas
   le meme cout qu'a depth 3, et on ne connait pas le taux d'erreur du RFP
   actuel (ses noeuds coupes ne sont jamais cherches). Seul un SPRT
   tranche.

## Etape 2 : integration et SPRT

Le modele (b) est integre tel quel, sans reseau : une regression par
profondeur (`src/engine/search/learned_prune.hpp`, poids generes par
`tools/learned_pruning/export.py` sur les 86M noeuds de l'etape 1, NMP
compris puisque le mecanisme est place avant lui). 75 multiplications par
noeud de decision, nps inchange.

- Parite C++ / Python : le dump v2 enregistre le logit du moteur,
  `export.py check` le recalcule depuis les poids. Ecart max 2.8e-6 sur
  89 888 noeuds.
- Seuil par defaut : T(depth) = 0.950 + 0.005 * depth, borne a 0.999
  (`learned_prune_t_base`, `learned_prune_t_depth`, `learned_prune_enabled`
  exposes au SPSA).
- `bench 13` : 2 002 833 -> 1 471 478 noeuds (-26.5 %). Les coupes se
  concentrent a depth 4-9 ; a depth 7 elles remplacent presque tout le NMP.
- Bench OpenBench : 16501 (16833 pour `main`, et avec
  `learned_prune_enabled=0`).

SPRT : test 53, `learned-pruning` @ 5bfdc66 contre `main` @ 9e9058b,
10+0.1, Threads=1 Hash=8, UHO_4060_v2, bornes [0, 3], alpha = beta = 0.05.

**Resultat du SPRT (test 53) : echec**, 1470 parties, +329 -595 =546
(41 %, environ -63 Elo), LLR -3.40.

## Etape 3 : pourquoi ca perd

**Balayage hors parties** (`tools/learned_pruning/sweep.sh`,
`sweep_report.py`) : 5000 positions de test jamais vues a l'entrainement.
Reference = mecanisme eteint a profondeur 14 ; courbe de base = eteint a
11, 12, 13 ; 20 configs a profondeur 12 (profondeur max d'application
2/4/6/64 x seuil). Critere : a noeuds egaux, s'accorder plus souvent avec
la reference que la courbe de base.

| config | noeuds / off12 | accord ref14 |
|---|---|---|
| off11 (eteint, un ply de moins) | 0.557 | 82.3 % |
| off12 | 1.000 | 87.9 % |
| off13 | 1.795 | 93.6 % |
| md64_950p5 (le SPRT) | 0.666 | 69.7 % |
| md64_990 | 0.898 | 76.7 % |
| md4_995 | 0.985 | 80.4 % |
| md2_995 (le plus prudent) | 0.996 | 84.1 % |
| controle : off12, Hash=4 | 0.989 | 87.2 % |

**Toutes les configs sont dominees** par "chercher un ply de moins". Meme
la plus prudente (seuil 0.995, depth <= 2) perd 3.8 points d'accord pour
0.4 % de noeuds economises, alors qu'une perturbation neutre de l'arbre
(Hash=4) n'en perd que 0.7.

**Mode ombre** (`learned_prune_enabled=2`, `precision.py`) : le mecanisme
calcule sa decision et remplit le cache d'eval, mais ne coupe pas ; le dump
enregistre z et le vrai resultat. Hypothese testee : le remplissage du cache
change la feature `eval_prev2` (connue a depth 8 dans 42 % des cas au lieu
de 18 %) et fausse le modele.

| depth | coupes | precision (eteint) | precision (ombre) |
|---|---|---|---|
| 1 | 25 % | 97.0 % | 97.1 % |
| 3 | 35 % | 97.8 % | 97.9 % |
| 5 | 40 % | 98.7 % | 98.7 % |
| 7 | 56 % | 99.5 % | 99.5 % |
| 9 | 38 % | 99.9 % | 99.9 % |

Hypothese rejetee : la precision tient en situation. **Le modele predit
bien ; ce sont les coupes qui coutent.**

## La lecon

**La precision par noeud est la mauvaise mesure.** Les noeuds dont le
resultat surprend -- ceux que les features statiques ne voient pas venir,
typiquement une tactique -- sont justement ceux qui decident du coup joue.
Un predicteur a 99.5 % se trompe sur ce 0.5 %-la, par construction, et
rendre une borne sans chercher les supprime sans rattrapage : la recherche
existe pour trouver les surprises. Les elagages a la main (RFP, NMP) se
trompent aussi, mais le NMP verifie par une recherche, et le RFP ne coupe
qu'avec des marges ou la surprise est rare.

**Suite possible** : reduire au lieu de couper. Une reduction garde une
recherche (moins profonde) du noeud, donc une chance de voir la surprise,
comme la re-recherche de la LMR. A valider par le meme balayage avant tout
SPRT.
