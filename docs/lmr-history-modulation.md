# LMR modulee par l'history : resultat negatif

Piste testee puis abandonnee, sur la branche `lmr-history-modulation`
(non fusionnee). Cette page garde la mesure, pour que personne ne la
retente en croyant qu'elle n'a jamais ete essayee.

## L'idee

Jusqu'ici la reduction LMR ne regarde que le rang du coup dans la liste :
deux coups calmes au meme rang sont reduits pareil, que l'un ait coupe cent
fois dans la partie et l'autre jamais. Les tables d'history savent les
distinguer. D'ou, comme chez Stockfish :

```cpp
r -= hist / HistDivisor;
```

Attendu : +10 a +20 Elo en auto-jeu a cadence courte.

## Le resultat

| test | comparaison | resultat |
|---|---|---|
| 40 | version brute (`HistDivisor` = 256) vs v5.4 | -3.5 +/- 6.6, 5108 parties |
| 43 | SPSA 4 parametres, distribution MULTIPLE | 32 004 parties |
| 44 | version tunee vs v5.4 | **+0.04 +/- 4.63**, 24 382 parties |

Zero. Le tune a recupere les ~3.5 Elo que la version brute perdait, sans
rien creer au-dela. Le mecanisme coute pourtant 7.7 % d'arbre a profondeur
13, donc il agit -- il n'achete simplement rien.

## Pourquoi : la LMR ne voit jamais un bon coup

Trois compteurs au site LMR, bench profondeur 13 :

```
appels = 594 241
hist positive =  0.7 %
terme nul     = 96.7 %
clamp bas     =  0.0 %
rang moyen    = 18.9
hist moyenne  =  -89
```

`MinMovesSearched = 5` : la LMR ne se declenche qu'a partir du 5e coup, en
pratique le 19e. Or **l'ordonnancement a deja place les coups a forte
history en tete**, donc ils sont recherches AVANT d'atteindre la LMR. Ce qui
arrive au site LMR, ce sont les rebuts : history nulle ou negative dans
99.3 % des cas.

Avec `hist <= 0`, le terme `r -= hist / D` **augmente** la reduction. Le
mecanisme ne recompense pas les bons coups, il punit un peu plus les
mauvais -- une variante de LMP deguisee. Et comme ces coups etaient deja
lourdement reduits et que la LMR re-cherche en cas de doute, le resultat ne
bouge pas.

Le clamp `std::clamp(r, 0, depth - MaxDepthReduction)` n'y est pour rien :
mesure a 0.0 %, il n'intervient jamais. C'etait l'autre hypothese, elle est
fausse.

## Ce qu'il faudrait pour que ca marche

Chez Stockfish le meme terme rapporte, pour deux raisons structurelles :

1. leur LMR se declenche des le 2e ou 3e coup, donc elle voit des coups a
   bonne history ;
2. leur `r` peut devenir **negatif**, ce qui ETEND un bon coup au lieu de
   simplement ne pas le reduire.

Donc ce n'est pas un meilleur `HistDivisor` qu'il faut chercher, mais faire
descendre `MinMovesSearched` et autoriser `r < 0`. Deux changements de
structure au coeur de l'elagage, chacun meritant son propre SPRT.

## La lecon generale

**Une heuristique qui module sur une information deja consommee en amont
arrive trop tard.** Cela se verifie AVANT d'ecrire la moindre ligne : il
suffit de compter ce qui arrive reellement au site qu'on veut modifier.
Trois compteurs et un bench, dix minutes. Ici, deux SPRT et un SPSA de
32 000 parties ont ete depenses pour l'apprendre.

A rapprocher de la regle du runbook SPSA : un parametre de reduction ne se
choisit pas au compte de noeuds -- moins de noeuds s'obtient trivialement
en elaguant plus. `HistDivisor = 256` avait ete choisi ainsi.
