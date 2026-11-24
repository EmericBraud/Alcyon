# Vitesse du NNUE : lecture de l'assembleur (branche `nnue-asm`)

Suite de `docs/vitesse-nnue.md`, avec le reseau actuel (SFNNv14 : Full_Threats
60 720 + HalfKAv2_hm 22 528, L1 1024, 8 tetes). Objectif : trouver dans le code
machine genere des chemins chauds ce que des intrinseques ecrits a la main
peuvent ameliorer, **a recherche strictement identique** (bench 16763, et meme
nombre de noeuds aux profondeurs 8, 11, 13 et 14).

## Methode

- Mac Apple M5 Pro (NEON + dotprod), GCC de Homebrew, `-march=native`.
- Profil : `sample` sur `alcyon bench 15`, tri par sommet de pile.
- Assembleur : `objdump -d --no-show-raw-insn --demangle build/alcyon`.
- Mesure : `alcyon bench 13` (1,67 M noeuds), paires A/B entrelacees,
  noeuds / temps CPU utilisateur (`/usr/bin/time -p`), mediane des rapports.
  La machine etait partagee (charge 15, un processus Java sur 10 coeurs) :
  le temps CPU est bien plus stable que le temps mur (ecart-type par paire
  ~3 % contre ~10 %). Les chiffres finaux ont ete pris machine calmee.
- La voie AVX2 ne peut pas etre chronometree ici. Elle est verifiee exacte
  par un harnais compile en x86_64 `-march=haswell` (clang d'Apple) et lance
  sous Rosetta, qui execute l'AVX2 : 20 000 accumulateurs aleatoires, l0, liste
  des groupes non nuls et sortie de L1 identiques a la reference scalaire.

## Profil de depart (sommets de pile, `main` 228ae9b)

| poste | part |
|---|---|
| `materialize` (mise a jour fusionnee de l'accumulateur) | 38 % |
| `get_result` (l0 + L1 creuse + L2 + sortie), deux perspectives | 20 % |
| collecte des menaces autour des cases touchees | 11 % |
| `threat_index` | 6 % |
| `filter_threats_diff` | 4 % |
| `Eval::eval` | 4 % |

## Goulots trouves dans l'assembleur

1. **l0 puis liste des groupes non nuls (`get_result`)**. Le code calculait l0
   (128 iterations de 8 valeurs, stockage de 8 octets), puis
   `get_raw_sparse` appelait `memcpy` pour recopier les 1024 octets de l0 dans
   un tableau de `uint32`, puis une boucle **scalaire de 256 iterations**
   (`ldr`, `strh`, `cmp`, `cinc`...) construisait la liste des groupes non
   nuls. Ce prologue coutait autant que le produit creux lui-meme.
2. **Mise a jour de l'accumulateur**. Blocs de 8 vecteurs de 8 lanes (64
   valeurs) : sur NEON, 24 des 32 registres restent inutilises, donc 16 passes
   sur la liste des lignes. Les lignes int8 se chargeaient par 8 octets
   (`ldr d` + `saddw`), soit deux fois plus de chargements que necessaire. Et
   chaque ligne de chaque bloc testait « menace (int8) ou HalfKA (int16) ? ».
3. **Couche L2 (32 x 64)**. Un produit scalaire par neurone, chacun termine par
   une reduction horizontale `addv` et un stockage scalaire : 32 reductions par
   evaluation.

## Changements retenus

| commit | changement | gain (bench 13, Mac) |
|---|---|---|
| edd7389 | l0 et liste des groupes non nuls en une passe vectorielle : le masque des groupes non nuls est tire des registres qui viennent d'etre calcules (`cmtst` + `addv` sur NEON, `cmpeq` + `movemask_ps` sur AVX2), une table donne les indices a ecrire (ecriture inconditionnelle, avance de `popcount`). Plus de `memcpy` ni de boucle scalaire. | **+4,2 a +4,8 %** |
| 1d24063 | mise a jour fusionnee specialisee NEON : 16 accumulateurs (blocs de 128 valeurs, deux fois moins de passes), lignes int8 chargees par 16 octets et elargies par `saddw`/`saddw2`, lignes converties en pointeurs et triees par largeur une fois par coup (plus de test par ligne et par bloc). L'arithmetique int16 est modulo 2^16, donc l'ordre des additions ne change rien au resultat. | **+11 %** |
| 8082f9d | couche L2 sur poids regroupes `[entree/4][neurone][4]` (ceux que L1 utilise deja) : 16 entrees chargees d'un coup, un `sdot` par lane, aucune reduction horizontale. NEON seulement. | **+1,0 a +1,4 %** |
| bf3ddf8 | gardes `if constexpr` pour les petits reseaux des tests unitaires | 0 |

**Cumul : +16 a +17 % de noeuds/s sur Mac** (16 paires, 1 053 k -> 1 217 k
noeuds/s CPU, ecart-type par paire 2,4 %), recherche identique, 124 tests
unitaires passes, cible `make` (SPSA) compilee, bench 16763.

Decomposition : le tri des lignes seul, compile en version generique sur ARM,
est neutre (0,998) ; le gain du 2e commit vient donc entierement des blocs
larges et des chargements de 16 octets, propres a NEON. Cette voie n'est
compilee que sur aarch64 ; x86 garde le code generique d'avant.

## Menaces : deltas directs au lieu de la difference avant/apres

Avant : pour chaque coup, `collect_move_scoped_features_both` collectait
toutes les menaces autour des cases touchees sur la position **avant** le
coup, puis **apres**, et `filter_threats_diff` en faisait la difference
(environ la moitie des features collectees s'annulaient).

Maintenant (`full_threats_delta.hpp`) : le coup est decompose en changements
elementaires (piece retiree de la case de depart, piece prise retiree,
piece posee a l'arrivee ; tour du roque), appliques un par un a un petit etat
local (occupation, bitboards, mailbox). Pour un changement sur la case s, les
features qui changent sont exactement :
(a) les menaces de la piece en s ; (b) les menaces sur la piece en s
(poussee de pion comprise, dans les deux sens) ; (c) les rayons decouverts :
chaque fou, tour ou dame qui voit s voit, s vide, la premiere piece au-dela.
Une feature retiree puis rajoutee dans le meme coup est annulee a
l'emission. Les coups de roi gardent le rafraichissement complet pour la
perspective du roi qui bouge ; l'autre perspective passe par les deltas.

Controle : compile avec `-DALCYON_THREAT_DELTA_CHECK`, chaque coup calcule
aussi l'ancienne difference et verifie l'egalite des ensembles retires et
ajoutes (arret `FATAL` sinon). Aucune divergence sur 6,8 M noeuds
(`bench 16`). Bench 16763, 124 tests passes.

Gain : **+13,4 %** de noeuds/s sur Mac (14 paires, bench 13, ecart-type par
paire 0,8 %, 1 308 k -> 1 494 k noeuds/s CPU). Ce changement n'a rien de
specifique a l'architecture : le gain devrait se retrouver sur x86.

**Cumul de la branche : environ +32 % sur Mac** (+16,5 % puis +13,4 %).

## Effet attendu sur les serveurs x86

- Le 1er commit s'applique aussi a AVX2 (verifie exact sous Rosetta) : meme
  suppression du `memcpy` et de la boucle scalaire de 256 iterations ;
  gain probablement du meme ordre (quelques %), **non mesure**.
- Les 2e et 3e commits ne changent rien sur x86 : en AVX2, 16 registres ymm,
  les blocs de 8 vecteurs sont deja le bon choix ; en AVX-512, un bloc fait
  deja 256 valeurs.
- A mesurer sur le serveur EPYC : nps (`bench 13`, binaires des deux
  branches), puis match court a recherche identique.

## Essais non retenus (mesures)

| essai | resultat |
|---|---|
| prechargement des lignes des deux perspectives avant d'appliquer la premiere | 0,99 (neutre) |
| prechargement de la rangee entiere (16 a 32 lignes de cache) au lieu de la premiere ligne seulement | **0,96** : sur le M5 la mise a jour n'est pas limitee par la memoire, les prechargements en trop coutent |
| boucle `sdot` de L1 deroulee par deux avec deux jeux d'accumulateurs (chaines de dependance deux fois plus courtes) | 0,996 (neutre) |
| `threat_index` par tables precalculees (paire -> base, rang de `to` dans les attaques de `from`) au lieu du masque d'attaques et du `popcount` recalcules | 0,998 sur 16 paires (neutre) |
| tri des lignes seul (version generique `stdx`, blocs de 8) | 0,998 (neutre) |

## Ce qui reste (profil apres les changements)

| poste | part |
|---|---|
| mise a jour de l'accumulateur (`fused_split`) | ~29 % |
| `get_result` | ~18 % |
| collecte des menaces + `threat_index` + `filter_threats_diff` | ~20 % |

- La mise a jour est maintenant limitee par le calcul (2 x ~10 lignes de 1 a
  2 Ko par coup) ; le seul levier restant est d'en lire moins (cache par case
  de roi pour les rafraichissements, voir `docs/vitesse-nnue.md`).
- Les menaces sont maintenant produites par deltas directs (voir plus haut) ;
  `threat_index` avec des tables reste neutre.
- Sur x86, le prechargement pourrait se comporter autrement (latence memoire
  plus grande sur EPYC) : a re-essayer la-bas avant de l'ecarter.
