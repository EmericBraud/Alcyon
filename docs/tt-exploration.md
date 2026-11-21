# Table de transposition : utilisation reelle et pistes

Branche `tt-explore`. Question de depart : toutes les entrees de la TT
servent-elles autant ? Une TT hierarchique (petit niveau tres rapide, grand
niveau) ou une meilleure exploitation des entrees peu utilisees peut-elle
aider ?

## Structure actuelle

Paquets de 4 entrees de 16 octets (une ligne de cache de 64 o), cle complete
64 bits, donnees XOR-ees avec la cle. Generation incrementee a chaque `go`.
Remplacement : meme cle -> ecrase si plus profond ou generation ancienne ;
sinon victime = plus petite priorite `profondeur + 100 si generation
courante`.

## Mesure (`-DALCYON_TT_STATS`, `tools/tt/tt_measure.py`)

Parties rejouees coup par coup, un moteur par partie (TT conservee), pendule
simulee, Hash 16 Mo, Threads 1.

| | 10+0.1 (120 parties) | 60+0.6 (60 parties) |
|---|---|---|
| lectures | 1,2 G | 3,0 G |
| cle trouvee / coupure | 26,6 % / 10,8 % | 30,5 % / 13,6 % |
| ecritures : place vide | 34,6 % | 7,6 % |
| ecritures : meme cle | 38,0 % | 33,9 % |
| ecritures : eviction | 22,6 % | 50,8 % |
| ecritures refusees | 4,9 % | 7,7 % |
| evincees d'un coup precedent | 100 % | 99,6 % |
| lectures par emplacement (0..3) | 27 / 26 / 24 / 22 % | 28 / 26 / 24 / 22 % |

Entrees evincees a 60+0.6, part jamais relue / relue 8 fois ou plus :

| profondeur | n | jamais lue | lue >= 8 fois |
|---|---|---|---|
| <= 0 | 186 M | 79,7 % | 0,6 % |
| 1-2 | 129 M | 68,0 % | 1,7 % |
| 3-5 | 83 M | 51,3 % | 4,2 % |
| 6-9 | 22 M | 23,9 % | 13,0 % |
| 10+ | 2,2 M | 5,6 % | 48,7 % |

## Lecture

1. A 10+0.1 avec 16 Mo la TT n'est jamais pleine pendant un coup : aucune
   entree du coup en cours n'est evincee. La capacite n'y est pas un
   probleme.
2. Les entrees sont tres inegales : l'immense majorite des ecritures
   (profondeur <= 2) ne sert jamais ; une petite fraction d'entrees
   profondes concentre la reutilisation.
3. Sous pression (60+0.6), la politique evince presque exclusivement les
   entrees des coups precedents, meme profondes : le bonus +100 de la
   generation courante fait qu'une entree neuve de profondeur 0 chasse une
   entree de profondeur 20 du coup d'avant, alors que les positions reviennent
   deux demi-coups plus tard et que les entrees profondes evincees avaient
   ete lues 8 fois ou plus dans ~49 % des cas.
4. Les 4 emplacements du paquet sont utilises a peu pres egalement en partie
   (le desequilibre 97 % / 3 % vu sur une recherche isolee venait d'une TT
   presque vide).

## Pistes

1. Remplacement par age relatif (Stockfish) : priorite = profondeur -
   k x (coups depuis l'ecriture) ; `tt_replace=1`, `tt_age_weight`.
2. Garde des entrees peu profondes : une entree de profondeur <= 0 n'evince
   qu'une place vide ou une entree d'un coup precedent ; `tt_shallow_guard`.
3. Entrees compactes de 8 octets (cle partielle 16 bits) : 8 par paquet,
   capacite doublee a cadence longue.
4. Hierarchie : petit niveau tenant dans le cache du processeur pour les
   entrees peu profondes et tres lues, grande table pour les profondes.

Tests : 10+0.1 avec Hash 2 Mo (meme rapport noeuds par coup / capacite qu'une
cadence longue avec 16 Mo, mais rapide).

| Variante | Elo | Parties |
|---|---|---|
| age relatif, k = 4 | (en cours) | 500 |
| garde des entrees peu profondes | (en cours) | 500 |
