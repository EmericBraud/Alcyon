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

- `depth`, `ply`
- `improving`, `eval(ply) - eval(ply-2)`
- TT : hit, type de borne, `tt_depth - depth`, `tt_score - static_eval`
  (borne ; c'est de l'information de recherche deja payee, que la
  position seule ne contient pas)
- type de noeud (cut / all) : l'issue que le parent attend
- `allow_null` (le parent vient d'un null move)
- materiel hors pions de chaque camp, nombre de coups legaux
- coup precedent : capture, echec, piece deplacee
- compteur des 50 coups, repetition possible : les seuls cas ou V depend
  vraiment du chemin

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

**Labels.** Mode de dump (build d'experimentation, `ALCYON_PRUNE_DUMP`) :
a un noeud sur N tire au hasard, au point de decision, on ecrit les
features, l'eval statique et beta, puis on laisse la recherche se derouler
normalement et on ecrit son resultat (fail-high ou non) et la taille du
sous-arbre. Enregistrements
de taille fixe dans un tampon statique ecrit par `fwrite` -- pas
d'allocation a l'execution.

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
