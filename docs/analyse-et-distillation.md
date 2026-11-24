# Analyse des parties et donnees de distillation (novembre 2026)

Ce document couvre, dans l'ordre chronologique :
1. l'analyse des parties de v6.2 (ou perd-il ?) ;
2. les essais de gestion du temps qui en ont decoule ;
3. les paliers d'accumulateur (suite de `eval-par-paliers.md`, branche `eval-tiers`) ;
4. la comparaison avec l'architecture NNUE actuelle de Stockfish ;
5. la generation d'un dataset de distillation Alcyon contre Stockfish actuel.

Scripts : `tools/distill/` (pipeline et analyses), `tools/gpu/tiers_multi.py`.
Donnees : `gpu-archive/` (local, hors git) et `gdrive:Alcyon/gpu-archive`.

## 1. Ou Alcyon perd-il ?

Corpus : 223 parties v6.2 contre SF8 (60+0.2, livre UHO, serveur EPYC). 

### Premiere lecture avec SF8 comme arbitre (abandonnee)

- Alcyon termine avec ~15 s de pendule inutilisees (SF8 ~1 s) et cherche 2 a 5 plies
  moins profond que SF8 a chaque phase.
- SF8 est plus faible qu'Alcyon : il ne peut pas servir d'arbitre fiable. Analyse refaite
  avec Stockfish actuel (commit `3e8fde7`, 22/11/2026).

### Arbitrage par Stockfish actuel (`referee.py`, `ref_an.py`)

Chaque coup d'Alcyon (17 036) evalue par SF avant/apres, 300 000 noeuds, en score
attendu (WDL de SF). Positions non decidees : 8 776.

| phase (materiel) | perte moy./coup | erreurs >= 10 % | part de la perte totale |
|---|---|---|---|
| >= 50 (sortie d'ouverture) | 2,24 % | 261 | 67 % |
| 30-49 | 1,23 % | 52 | 16 % |
| 12-29 | 0,45 % | 34 | 11 % |
| < 12 | 0,57 % | 13 | 6 % |

- Coups calmes : 5,0 % d'erreurs >= 10 %, contre 2,7 % (captures), 2,2 % (echecs).
- Positions equilibrees : 1,5 % d'erreurs ; avantage 0,75-0,95 : 13,9 % ; desavantage : 11,5 %.
- Les defaites partent de positions UHO plus difficiles (score attendu initial 0,34 contre 0,66
  pour les victoires).

### Recherche ou eval ? (`extract.py`, `rethink.py`, `rt_an.py`)

304 positions d'erreur sur coup calme rejouees par Alcyon avec 1, 4, 16 et 64 s.
Le bruit de l'arbitre (SF 300 k noeuds) est important : 33 des 150 coups rejoues a
l'identique sont rejuges acceptables. Sur les 117 erreurs confirmees :

| temps | profondeur | corrigees |
|---|---|---|
| 4 s | 15,8 | 14 % |
| 16 s | 18,6 | 26 % |
| 64 s | 22,5 | 43 % |

A profondeur 12, Alcyon juge son coup au moins aussi bon que celui de SF dans 68 % des
cas. Conclusion : erreurs surtout d'**eval**, la profondeur n'aide que lentement.
(Un rejugement a 10 M noeuds, `rejudge.py`, a ete prepare mais interrompu.)

## 2. Gestion du temps

Variantes contre v6.2, 10+0.1, 500 parties, meme binaire (bench 16763) :

| variante | Elo | pertes au temps |
|---|---|---|
| 100 % de l'increment | -24,4 +- 20,4 | 41 (toutes la variante) |
| 100 % + 1/22 de la pendule | -20,9 +- 22,2 | 72 (toutes la variante) |
| **80 % de l'increment** | **+14,6 +- 18,7** | 4 (toutes la variante) |

Les pertes des deux premieres variantes viennent des pertes au temps : sans reserve,
la pendule tombe pres de zero et le moindre depassement fait perdre.
Branche locale `tm-inc80` : 80 % + limite souple plafonnee a 50 % du budget restant
(contre les pertes au temps). **Non validee en parties** (validation lancee puis annulee).

## 3. Paliers d'accumulateur : plusieurs granularites (`tiers_multi.py`)

Une petite tete par taille k (entrees de la 1re couche par perspective, prefixes emboites
dans l'ordre d'importance `ft.perm`), entrainees ensemble. **k = 512 = tout l'accumulateur**
(produit par paires : 1024 colonnes -> 512 entrees) ; k = 256 = 50 % des colonnes, etc.

Ecart RMS a la tete complete (cp, |eval| < 600), v5 fige :

| k (part de l'accumulateur) | 16 (3 %) | 32 | 64 (12,5 %) | 128 (25 %) | 256 (50 %) | 512 (100 %) |
|---|---|---|---|---|---|---|
| tetes aleatoires | 195 | 186 | 171 | 146 | 109 | 28,5 |
| tetes initialisees depuis la complete | 195 | 187 | 172 | 147 | 107 | 7 |

- Les 28,5 cp de k = 512 etaient un defaut de methode (tete partie de zero) : corrige par
  l'initialisation depuis la tete complete (7 cp residuels).
- Pour k <= 256, la perte est reelle : v5 repartit l'information sur tout l'accumulateur.
- Adapter v5 (fine-tuning conjoint) n'aide presque pas (-10 % d'ecart pour -1,5 % sur la
  tete complete) ; poids des petites tetes trop forts = tete complete degradee de 5 %.
- Adam deplace les poids rares meme a lr faible : lr modele 2e-6 necessaire pour ne pas
  degrader la tete complete.
- Gain attendu d'un palier 256 utilise avec confirmation (marge ~300) : quelques Elo au mieux.
  Seule piste : reseau entraine de zero avec les paliers dans la perte. Non poursuivi.

## 4. Architecture NNUE de Stockfish (22/11/2026) contre Alcyon

| | Alcyon (v5) | Stockfish |
|---|---|---|
| accumulateur / couches denses | 1024, 32 -> 32 -> 1 | identique |
| menaces | Full_Threats 60 720 | 59 808 |
| piece-case | HalfKAv2_hm | K32H3 (3 blocs selon tours/dames) |
| paires de pions | - | PP_3Wide |
| PSQT | 8 compartiments | supprimee |
| tetes | 8 (nombre de pieces) | 192 |
| petit reseau | - | supprime |

Gains publies (LTC, calcules des pentanomiaux des commits) : v14 +1,4, v15 (recette) +3,3,
v16 paires de pions +1,2 (VVLTC +2,3), v17 sans PSQT ~0, v18 K32Q2 + 32 tetes +3,6,
v19 K32H3 +2,7, v20 192 tetes +2,1. Environ +10 a 12 Elo LTC au total pour ce qui nous manque.

## 5. Dataset de distillation

### Idee

Le fine-tuning de v5 sur test80-2024 a ete neutre (`reentrainement-v5.md`). Hypothese :
apprendre sur la distribution d'Alcyon (ses positions, celles qui suivent ses erreurs),
etiquetee par un professeur plus fort.

### Echelle des scores (`calib.py`)

Stockfish modifie (`sfraw`) : `UCIEngine::to_cp` renvoie la valeur interne brute.
Sur 600 positions de parties : ecart-type SF brut / eval statique v5 = **0,99**, correlation
0,83 -> meme echelle que les sorties de v5, utilisable telle quelle comme cible.

### Generateur (`gen3.py`)

Parties Alcyon (profondeur 7, 1 thread) contre SF actuel brut, depuis UHO_4060_v2,
couleur au hasard, 64 workers.
- Une position sur 3 tiree au sort **avant** etiquetage : SF y cherche a profondeur 9
  (score + coup = etiquette) ; les autres coups de SF a profondeur 5.
- Feuille de PV : pour un coup d'Alcyon sur 3, sa PV est jouee jusqu'a la feuille ; on
  enregistre l'eval statique d'Alcyon (commande `eval`) et SF profondeur 9 sur la feuille.
  Controle : eval statique de la feuille = score racine d'Alcyon (398 = 398).
- Adjudication : |score SF| >= 1500 sur 3 coups SF -> victoire ; |score| <= 15 sur 10 coups
  apres le ply 80 -> nulle (SF a profondeur 9 ne convertit pas toujours : roi qui tourne au
  lieu de promouvoir).
- Morceaux de 50 parties, ecrits en `.tmp` puis renommes : `.plain.gz` (entrainement),
  `.diag.gz` (eval Alcyon racine / SF), `.leaf.gz` (feuilles).

Historique : run1 (gen2 sans feuilles, Alcyon prof. 8, 6,16 M positions), run2 (gen2 avec
feuilles), run3 (gen3). Pieges : Alcyon demarre par defaut avec 64 threads (forcer
Threads=1) ; `is_game_over(claim_draw=True)` de python-chess tres couteux.

Debit (c8g.16xlarge, 64 coeurs) : gen2 ~590 positions conservees/s, gen3 ~1 020/s (+ ~930
feuilles/s). Repartition CPU avant gen3 : Alcyon 50 %, SF 27 %, Python 22 %.

`leafmine.py` : feuilles a posteriori pour run1, sur les 2,0 M coups ou Alcyon et SF
divergent de >= 200 (1,76 M feuilles, ~2 400/s).

### Ce que montrent les feuilles (`leaf_wp.py`, `sf_static.py`)

Ecart en points de probabilite de gain (sigmoide v/340), feuilles calmes, eval statique
contre recherche SF profondeur 9 :

| \|P\| SF | Alcyon (tous coups) | SF statique (tous coups) | Alcyon (desaccords) | SF statique (desaccords) |
|---|---|---|---|---|
| 40-60 % | 8,2 | 6,0 | 17,2 | 11,8 |
| 30-40 / 60-70 % | 9,1 | 9,0 | 13,3 | 12,2 |
| 20-30 / 70-80 % | 9,5 | 9,5 | 11,6 | 11,5 |
| 10-20 / 80-90 % | 13,6 | 11,9 | 11,9 | 10,6 |
| < 10 / > 90 % | 16,4 | 10,5 | 12,5 | 7,1 |
| ensemble | 11,0 | 8,6 | 13,2 | 9,4 |

- ~3/4 de l'ecart est irreductible (l'eval de SF l'a aussi face a sa propre recherche).
- La marge utile est dans les positions **equilibrees** (8,2 -> 6,0 ; 17,2 -> 11,8 en desaccord).
- Les gros ecarts bruts des positions decidees sont de la saturation d'echelle : negligeables
  apres la sigmoide de la perte.
- Biais faible : Alcyon legerement pessimiste pour le camp au trait (-2,4 points).

### Construction (`build_ds.py`, `auto_ds.sh`)

Filtres : pas en echec ; coup SF ni capture ni promotion ; P(SF) entre 20 et 80 % ;
dedoublonnage par FEN ; saut aleatoire 1/3 pour run1/run2 (run3 deja echantillonne) ;
feuilles de desaccord plafonnees a 25 % (ancrage : majorite de positions ou Alcyon etait juste).
Conversion en binpack : branche `tools` de Stockfish (`stockfish convert x.plain x.binpack`),
verifiee par aller-retour identique.

Dataset final (`gpu-archive/aws-ds/final/`, 5 binpacks, 328 Mo) :

| source | positions |
|---|---|
| parties run1 | 1 173 671 |
| parties run2 | 1 731 964 |
| parties run3 | 2 755 717 |
| feuilles run2 | 1 470 240 |
| feuilles run3 | 2 335 838 |
| feuilles desaccord run1 | 175 513 |
| **total** | **9 642 943** |

### Sauvegarde (`backup.sh`, `pack.py`)

Boucle sur le Mac (tmux + caffeinate) toutes les 15 min : rsync instance -> Mac, regroupement
en archives tar par serie et par heure (Drive limite le nombre de fichiers/s, pas le volume),
rclone vers `gdrive:Alcyon/gpu-archive`. Le remote `gdrive` doit avoir le scope `drive`
(il etait en `drive.readonly`). Etat final verifie : `rclone check` 0 difference.

### Dataset final de fine-tuning (`make_mix.py`, `binpack_chunks.py`)

Un binpack est une suite de chunks independants ("BINP" + taille) : on peut en extraire et en
concatener sans decoder. Les chunks test80 contiennent ~400 k positions d'affilee (parties
consecutives) ; le melange est donc fait au niveau des positions (binpack -> plain -> melange ->
binpack), en ~1 min sur le Mac.

| fichier | contenu | positions |
|---|---|---|
| `train_mix.binpack` (1,06 Go) | 30 % distillation + 70 % test80-2024-03-mar v6 (chunks au hasard) | 31 130 660 |
| `val_distill.binpack` | 10 chunks de distillation mis de cote | 303 745 |
| `val_feb.binpack` | 3 chunks de test80-2024-02-feb v6 (source de la perte de reference de v5) | 1 206 406 |

Dans `gpu-archive/train/` et `gdrive:Alcyon/gpu-archive/train`.

## 6. Fine-tunings de v5 sur le dataset de distillation (RTX 5090, vast.ai)

lr 3e-5, gamma 0,8, ranger21. Validations : `val_distill` (positions de distillation mises de cote)
et `val_feb` (test80 fevrier). Matchs contre v5 : meme binaire v6.2, 10+0.1, 400 parties.

| variante | val_distill | val_feb | Elo contre v5 (epoques testees) |
|---|---|---|---|
| v5 | 0,00753 | 0,00387 | - |
| distillation seule (9,3 M), lambda 1 | 0,00295 | 0,00585 (+51 %) | -107,5 / -101,8 / -107,5 |
| melange 31 M (30 % distillation), lambda 1 | 0,00367 | 0,00381 | +2,6 / -6,1 / -6,1 |
| melange 31 M, lambda 0,6 | - | 0,00381 | -12,2 / +0,9 |
| melange 31 M, lambda 0,6, resultats de la distillation re-tires selon le score SF | - | 0,00397* | -9,6 / -31,4 |

\* validation a lambda 0,6, non comparable directement.

- La distillation seule sur-corrige (derive forte sur les donnees generales).
- Le melange corrige la moitie de l'erreur sur nos positions sans deriver, mais aucun gain en parties.
- Re-tirage des resultats (`mix_server.py --resample`) : pour p >= 0,5, gain avec proba 2p-1 sinon
  nulle (symetrique) -> avec lambda 0,6, cible moyenne = score SF sur ces positions. Pas de gain.
- Incident : disque plein (89 Go) -> un entrainement planta et un melange fut tronque ; refaits.

## 7. Mesure privee : reseau Stockfish v14 dans Alcyon

Question : combien vaut une eval de niveau Stockfish **a architecture identique** ?
Le feature transformer de SFNNv14 (`nn-7bf13f9655c8.nnue`, avril 2026) est identique a celui de v5
(meme empreinte : Full_Threats 60720 + HalfKAv2_hm, L1 1024, 8 PSQT, 8 piles) ; seule la tete dense
differe (SF : fc0 31+1 raccourci, carre/ReLU sur 31, fc1 62->32, fc2 32->1 ; v5 : sortie sur la
concatenation l1+l2, 128 entrees). Conversion du fichier (LEB128, ordre des PSQT) et tete SF
transcrite sous un drapeau de compilation, sur une branche **locale non publiee** (`sf14-test`) :
les reseaux Stockfish sont sous GPLv3, Alcyon sous MIT -> mesure uniquement, rien d'integre.

| | Elo contre v5 (meme recherche v6.2, 10+0.1, 400 parties) |
|---|---|
| reseau Stockfish v14 dans Alcyon | **+205,0 +- 27,6** (243 V / 31 D / 126 N, 0 perte au temps) |

Conclusion : a recherche et architecture identiques, la qualite d'entrainement du reseau vaut
~200 Elo. Le levier principal d'Alcyon est un entrainement de reseau serieux (donnees massives,
recette Stockfish, entrainement long), pas le fine-tuning de v5 ni l'architecture.
(Les evals SF sont ~2x plus petites que v5 dans les positions decidees : marges de recherche
reglees pour v5, donc +205 est plutot une borne basse.)

## Suite

1. Fine-tuning de v5 sur un melange dataset de distillation / donnees d'origine (GPU ~1 h).
2. Match contre v5 ; si positif, generer davantage (objectif initial 20 M).
3. Valider la gestion du temps 80 % (`tm-inc80`).
4. Plus loin : architecture actuelle de Stockfish (K32H3, paires de pions, 192 tetes), entrainee de zero.
