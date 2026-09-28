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

Dans un noeud non-PV la fenetre est nulle (`beta = alpha + 1`), donc la
recherche n'a que deux issues : fail-high (`score >= beta`) ou fail-low
(`score <= alpha`). Une seule sortie suffit :

```
p = P(fail-high | features du noeud)       // sigmoide

si p >= T(depth)       -> return beta       // comme RFP / NMP
si p <= 1 - T(depth)   -> return alpha      // comme razoring
sinon                  -> recherche normale

T(depth) = clamp(T0 + T1 * depth, 0.5, 1)  // T1 > 0 : plus exigeant en haut
```

Le reseau generalise donc RFP, razoring et NMP d'un seul coup, avec le
meme mecanisme des deux cotes. `T0` et `T1` sont les seuls parametres
laisses au SPSA -- deux, pas des milliers (voir "Ce qu'on ne fait pas").

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

Scalaires du noeud (tous deja disponibles a l'endroit de la decision) :

- `depth`, `ply`
- `static_eval - beta` (borne), `improving`, `eval(ply) - eval(ply-2)`
- TT : hit, type de borne, `tt_depth - depth`, `tt_score - beta` (borne),
  `|tt_score - static_eval|` (complexite)
- type de noeud (cut / all), `allow_null` (le parent vient d'un null move)
- materiel hors pions de chaque camp, nombre de coups legaux
- coup precedent : capture, echec, piece deplacee

Plus l'accumulateur NNUE du noeud (L1 = 1024 par camp), qui encode la
position -- tension tactique comprise -- et qui est deja calcule des que
le noeud evalue. A mesurer : combien de noeuds atteignent la decision sans
avoir evalue (cout d'une mise a jour d'accumulateur en plus).

Architecture de depart : accumulateur -> 16 (int8, SIMD) concatene aux
scalaires -> 16 -> 1. De l'ordre de 17k MAC, soit quelques dizaines de ns :
a comparer au nps (environ 600k en bench, donc ~1.7 us par noeud).

## Donnees et entrainement

**Labels.** Mode de dump (build d'experimentation, `ALCYON_PRUNE_DUMP`) :
a un noeud sur N tire au hasard, au point de decision, on ecrit les
features, puis on laisse la recherche se derouler normalement et on ecrit
son resultat (fail-high ou non) et la taille du sous-arbre. Enregistrements
de taille fixe dans un tampon statique ecrit par `fwrite` -- pas
d'allocation a l'execution.

**Positions.** Les binpacks existants, a une profondeur fixe moderee, ou
les parties d'auto-jeu d'OpenBench.

**Entrainement** sur le serveur GPU, a cote de `training/nnue-pytorch` :
entropie croisee binaire. Les erreurs n'ont pas toutes le meme prix : on
pondere par la taille du sous-arbre (ce qu'une coupe juste economise, ce
qu'une coupe fausse rate). Reseau fige ensuite, quantifie en int8.

**Decalage de distribution.** Une fois le reseau actif, il coupe aussi des
descendants, donc l'arbre change. Re-dumper avec le reseau actif et
re-entrainer une ou deux fois.

## Etapes

0. **Compter ce qui arrive au point de decision.** Fait, voir ci-dessous.
   C'est la lecon de `docs/lmr-history-modulation.md` : on mesure avant
   d'ecrire le mecanisme.
1. **Essai sans reseau** : le trou mis en evidence a l'etape 0 (RFP a
   profondeur >= 7 avec une grosse marge) se teste avec deux parametres
   existants. SPRT. Si ca rapporte, c'est la nouvelle ligne de base.
2. **Dump + regression logistique hors ligne.** Critere de poursuite :
   a chaque profondeur, part du sous-arbre coupable avec une precision
   >= T(depth), pour (a) `static_eval - beta` seule, (b) toutes les
   features scalaires, (c) scalaires + accumulateur. Si (b) et (c) ne
   battent pas nettement (a), on s'arrete la : le reseau n'aurait rien a
   dire que l'elagage actuel ne dit deja.
3. **Integration** du reseau (inference int8, accumulateur, features),
   derriere une option UCI eteinte par defaut. Verifier l'impact nps.
4. **Recompense dense hors ligne** pour choisir `T0`, `T1` sans jouer de
   parties : sur un jeu de positions fixes, comparer
   `search(d, reseau actif)` a `search(d, reseau eteint)` --
   meme meilleur coup ? combien de noeuds en moins ?
5. **SPSA sur `T0`, `T1`**, puis **SPRT** contre la version de base, et
   un controle contre Stockfish 8 (pas seulement de l'auto-jeu).
6. Plus tard, hors perimetre : tete de reduction, qsearch, affinage des
   poids par la recompense dense (ES).

## Ce qu'on ne fait pas, et pourquoi

**Pas d'apprentissage par renforcement sur le resultat des parties.** Une
partie represente environ 10^8 decisions d'elagage pour un seul resultat,
et le SPSA a deja besoin de ~30 000 parties pour 4 parametres (tune 43).
Sur des milliers de poids, avec ~23 000 parties par jour, ca ne converge
pas. Le resultat des parties ne sert qu'aux 2 parametres de l'etape 5.

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
   TT doivent faire la difference. L'etape 2 le mesure avant d'integrer
   quoi que ce soit.
3. **Un trou simple a depth >= 7 :** le RFP s'arrete a depth 6. Au-dessus,
   les noeuds cherches avec `eval - beta >= 400` echouent haut a 98-99 %,
   et le NMP n'en rattrape qu'une partie. C'est 5 a 6.5 % du sous-arbre a
   ces profondeurs, a tester d'abord sans reseau (etape 1). 98.8 % n'est
   pas 100 % : a haute profondeur l'erreur coute cher, c'est au SPRT de
   trancher.
4. **Cote fail-low**, le meilleur bucket (`< -400`, depth 3-6) reste a
   11-14 % de fail-high : le razoring actuel n'a pas de marge evidente a
   recuperer avec l'eval seule.
