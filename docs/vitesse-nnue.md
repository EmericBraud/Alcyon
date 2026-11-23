# Vitesse du NNUE (branche `nnue-speed`)

Objectif fixe : +30 a 50 % de noeuds par seconde, sans perte de precision,
chaque etape validee (bench identique, puis parties).

## Point de depart : ou part le temps

Profil d'echantillonnage (macOS `sample`, thread de recherche, v6.1) :
l'evaluation prend ~89 % du temps (`docs/eval-par-paliers.md`, mesure 1).

| poste | part |
|---|---|
| mise a jour incrementale de l'accumulateur | 44 % |
| couches denses | 27 % |
| features de menaces | 11 % |
| copies d'accumulateurs (pile d'annulation) | 7 % |
| tout le reste (recherche, movegen, TT) | 11 % |

## Etapes (toutes a resultat identique : bench 16763 a chaque etape)

Mesures : 8 positions UHO, 2 s chacune, 1 thread, moyenne de deux passes.

| etape | idee | gain mesure |
|---|---|---|
| 1. mise a jour fusionnee (`fused_acc`) | chaque feature modifiee relisait et reecrivait les 2 Ko de l'accumulateur ; on traite l'accumulateur par blocs de 64 valeurs gardes en registres et on y applique toutes les lignes retirees puis ajoutees du coup : une seule lecture et une seule ecriture par coup | +9 % |
| 2. premiere couche creuse (`sparse_l1`) | l'entree de la 1re couche (1024 octets, produit par paires) n'a que **11 % d'octets non nuls** ; on construit sans branchement la liste des groupes de 4 octets non nuls, puis 4 entrees x 4 neurones par instruction (`sdot` ARM, `maddubs`+`madd` AVX2) | +16 % |
| 3. eval statique memorisee en TT (`tt_eval_cache`) | branche `tt-eval` : l'eval NNUE brute est stockee dans les 16 bits hauts du champ cle (cle verifiee sur 48 bits, entree toujours 16 o) et relue au lieu de recalculer | +4 % |
| 4. pile par echange de tampons | empiler echangeait un `memcpy` de 4 Ko par ply ; on echange maintenant deux pointeurs : l'ancien etat reste dans la pile, la mise a jour fusionnee le lit et ecrit le nouveau dans le tampon libre | +10 % env. |

Faux departs documentes :
- premiere version creuse octet par octet : **-7 %** (elargissement et
  multiplication scalaires par neurone) ;
- version par groupes avec un test `if (groupe nul)` : **-7 %** encore : 256
  branchements imprevisibles par eval. La liste sans branchement (ecriture
  inconditionnelle, avance conditionnelle de l'indice) a tout change.

## Resultat (binaires release, Mac Apple Silicon)

| binaire | noeuds/s | bench |
|---|---|---|
| v6.1 (`main`) | 744 000 | 16763 |
| **`nnue-speed`** | **1 073 000** | 16763 |

**+44 %**, recherche strictement identique.

Sur le serveur x86 (AMD EPYC 7543, AVX2), version sans l'etape 4 : 371 k ->
469 k noeuds/s (+26 %), et en parties (10+0.1, 491 parties, meme binaire,
optimisations desactivees pour la base) : **+35,6 +- 20,7 Elo**.

## Validation finale en parties

Serveur x86 (AMD EPYC 7543), 10+0.1, Hash 16, 1 thread, livre UHO,
concurrence 50, deux binaires distincts compiles depuis leurs branches
(bench 16763 pour les deux) :

| match | Elo | parties |
|---|---|---|
| nnue-speed (etapes 1-3) vs meme binaire sans les optimisations | +34,9 +- 20,3 | 500 |
| **nnue-speed (etapes 1-4) vs v6.1 (`main`)** | **+46,8 +- 19,2** | 500 (169 V / 102 D / 229 N, 0 perte au temps) |

Duree de chaque match : ~10 min.

## Ce qui reste (profil apres les 4 etapes)

| poste | part |
|---|---|
| mise a jour de l'accumulateur (une passe par coup, ~10 lignes de poids) | 37 % |
| couches denses (l0 + couche creuse + 2 petites couches) | 22 % |
| features de menaces (collecte avant/apres, difference, index) | ~20 % |
| copies d'accumulateurs | < 1 % |

- Rafraichissements apres un coup de roi : 2 % des mises a jour mais 10 % des
  lignes lues (54 lignes contre 10) ; un cache par case de roi ("Finny
  tables") gagnerait ~3 % du temps total.
- Les menaces sont deja recalculees seulement autour des cases touchees ;
  aller plus loin demanderait de les maintenir de facon incrementale.
