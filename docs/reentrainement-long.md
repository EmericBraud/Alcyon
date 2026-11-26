# Réentraînement long de v5 sur les données Leela filtrées (24-25/11/2026)

Suite de `analyse-et-distillation.md` (section 7) : le réseau SF v14 chargé dans Alcyon
battait v5 de +205 Elo à recherche et architecture identiques. Le levier était donc
l'entraînement du réseau, pas la recherche ni l'architecture. Budget fixé : 2 jours de GPU.

## Diagnostic de départ

- L'entraînement d'origine de v5 s'est arrêté à l'époque 15, avec un taux d'apprentissage
  encore à 88 % de sa valeur initiale (`strength-measurements.md`, v5.5).
- Données d'origine (selon l'utilisateur) : `test80-2024-01-jan-2tb7p.tar.zst`, c'est-à-dire
  les données Leela brutes d'un seul mois, **non filtrées**. Les scores sont des évaluations
  de Leela après recherche, pas l'éval statique de Stockfish comme l'écrit
  `strength-measurements.md`.
- Le fine-tuning v5.5 n'avait fait que 9 + 4 époques, et les fine-tunings de distillation
  (section 6 de `analyse-et-distillation.md`) étaient neutres.

## Données

Binpacks filtrés de linrock (Hugging Face, `linrock/test80-2024` et `linrock/test80-2023`),
format `min-v2.v6`, ceux qu'utilise Stockfish :
test80 2024 mars, avril, mai et juin, et 2023 novembre, soit environ 52 Go. Janvier et
février 2024 sont écartés : v5 en a vu une partie, et février sert de validation
(`val_feb`, 300 000 positions).

## Recette

Run long (`tools/distill/run_long.sh`), à partir de v5 :

- ranger21, lr 2e-4, gamma 0,995 par époque de 100 M positions, lot 16 384 ;
- lambda de 1,0 à 0,75 (linéaire sur 400 époques) ;
- filtres par défaut de nnue-pytorch (échecs et captures sautés, saut aléatoire) ;
- checkpoint toutes les 10 époques, exporté et joué contre v5 (400 parties, 10+0.1, même
  binaire v6.2, `tools/distill/watch_long.sh`).

Matériel : RTX 5090, 64 cœurs, vast.ai, 0,68 $/h. Débit : 16 it/s (6,3 min par époque),
descendu à 14 it/s après une vingtaine d'heures (cause non cherchée). Le GPU est à 95-100 %
d'occupation mais à ~300 W sur 575 : ce sont les noyaux creux (features de menaces) qui
limitent, pas le chargeur (passer de 24 à 48 workers et TF32 : aucun gain).

Sauvegarde : toutes les 30 min depuis le serveur vers `gdrive:Alcyon-serveur/long`
(`tools/distill/backup_long.sh`, jeton rclone limité à `drive.file`).

## Résultats du run long (contre v5, 400 parties chacun)

| époque | lr | Elo contre v5 | val loss |
|---|---|---|---|
| 9 | 1,91e-4 | +51,6 ± 25,0 | 0,00333 |
| 19 | 1,82e-4 | +82,3 ± 26,1 | 0,00329 |
| 29 | 1,73e-4 | +75,9 ± 25,2 | 0,00325 |
| 39 | 1,65e-4 | +105,6 ± 25,2 | 0,00319 |
| 49 | 1,56e-4 | +98,1 ± 26,6 | 0,00318 |
| 59 | 1,49e-4 | +82,3 ± 25,4 | 0,00320 |
| 69 | 1,42e-4 | +124,0 ± 25,9 | 0,00321 |
| 79 | 1,35e-4 | +115,2 ± 25,5 | 0,00319 |
| 89 | 1,28e-4 | +117,2 ± 27,5 | 0,00319 |
| 99 | 1,22e-4 | +95,3 ± 26,0 | 0,00320 |
| 109 | 1,16e-4 | +110,4 ± 28,8 | 0,00325 |
| 119 | 1,10e-4 | +110,4 ± 25,8 | 0,00324 |
| 129 | 1,05e-4 | +140,0 ± 27,6 | 0,00327 |
| 139 | 9,96e-5 | +113,3 ± 28,4 | 0,00324 |
| 149 | 9,48e-5 | +88,7 ± 25,0 | 0,00327 |
| 159 | 9,02e-5 | +125,0 ± 28,3 | 0,00326 |

Aucune perte au temps. Lecture :

- **L'essentiel du gain arrive très vite** : +52 après une heure, environ +100 après quatre.
- **Plateau à environ +114 de l'époque 69 à l'époque 159.** Les dix points ont un écart-type
  d'environ 14 Elo, exactement le bruit attendu d'un match de 400 parties (±27 à 95 %) : rien
  ne bouge réellement pendant ces 90 époques. Le +140 et le +89 sont du bruit.
- La val loss remonte légèrement après l'époque 100, mais lambda descend (0,91 à l'époque
  160) : la loss contient alors plus de résultats de parties, plus bruités, et n'est plus
  comparable d'un bout à l'autre du run. Seuls les matchs tranchent.

Le run a été arrêté à l'époque 163 (environ 20 h de GPU) au lieu des 43 h prévues.

## Sonde de recuit

À partir du checkpoint de l'époque 159 : 6 époques, lr 9e-5 × 0,7^k (environ 1e-5 à la fin),
lambda fixé à 1,0 (`tools/distill/anneal_probe.sh`). 38 minutes de GPU.

| réseau | Elo contre v5 | parties | V / D / N |
|---|---|---|---|
| **recuit A (lambda 1,0)** | **+123,5 ± 17,8** | 800 | 383 / 110 / 307 |

Val loss (lambda 1,0, donc comparable au début du run) : 0,00295, contre 0,00347 au départ.

Le recuit ne rapporte presque rien de mesurable : +123,5 contre environ +114 pour le
plateau, écart dans les marges. Contrairement à v5.5 (+41 par le seul recuit), le réseau de
l'époque 159 n'était pas loin de son minimum. Le plateau vient donc des données et de la
capacité atteignable par cette recette, pas d'un manque de recuit.

Une seconde sonde à lambda 0,90 était prévue pour mesurer l'effet de lambda ; elle a été
annulée. Entre 1,0 et 0,9, l'effet attendu (quelques Elo, par proportion avec le −22 de
lambda 0,6 en v5.5) est bien inférieur à la marge d'un match de 800 parties.

## Mesure contre Stockfish 8

Protocole du README (60+0.2, Threads 1, Hash 64, UHO_4060_v2), sur le serveur vast
(concurrence 30). v5 (v6.2) est rejoué sur la même machine comme référence, pour que l'écart
ne dépende pas du matériel.

| réseau | Elo contre SF8 | V / D / N | pertes au temps |
|---|---|---|---|
| **recuit A** | **+134,0 ± 24,2** | 204 / 57 / 139 | 0 |
| v5 (v6.2, référence) | +25,2 ± 25,1 | 153 / 124 / 123 | 0 |

Écart A − v5 contre SF8 : **+109 Elo**, cohérent avec le +123,5 mesuré en direct contre v5.
v5 fait ici +25 contre +52,4 lors de la mesure précédente sur le Mac (écart dans les marges,
ou effet du matériel) : c'est pourquoi la référence est rejouée sur la même machine.

Estimation CCRL (indicative) : SF8 ≈ 3359 + 134 ≈ 3490, ou ancienne estimation de v6.2
(≈ 3411) + 109 ≈ 3520. Soit environ **3490-3520 CCRL**.

## Coût

Environ 23 h de serveur (téléchargement, run long, recuit, mesures), soit environ 15 $.

## Suites possibles

- Nouveau réglage SPSA de la recherche pour ce réseau (marges réglées pour v5).
- Paliers d'accumulateur (`eval-par-paliers.md`, branche `eval-tiers`) : la mise à jour
  partielle donnait +62 % de nœuds/s, l'échec venait uniquement de la précision (107 cp
  après 11 minutes de fine-tuning Matryoshka). Un entraînement Matryoshka long à partir de ce
  réseau est le test naturel.
- Plus de données (autres mois de test80, `leela96`, `dfrc`) pour dépasser le plateau.

## Où l'erreur reste élevée (analyse du 25/11)

But : savoir quelles positions cibler si on ajoute des données. Fait sur le serveur
(`tools/distill/dump_binpack.cpp`, `err_eval.py`, `err_buckets.py`, 2 min 30 sur 64 cœurs).

Méthode : 735 684 positions calmes de `val_feb` (sans échec, coup joué ni capture ni
promotion, comme le filtre d'entraînement). Chaque position est évaluée par le moteur avec
v5, le nouveau réseau (recuit A) et le réseau SF v14 (branche locale `sf14-test`). Erreur =
écart au carré en probabilité de gain, sigmoïde(eval/400), entre le réseau et le score Leela,
× 10⁴. Une échelle par réseau est ajustée (v5 0,82, A 0,98, SF v14 0,92) : les evals de v5
sont trop grandes d'environ 20 % par rapport aux scores Leela.

Ensemble : v5 58,6, A 50,7, SF v14 48,2. La colonne « part du reste » donne la part de
l'écart A − SF v14 que porte chaque catégorie (une part négative : A fait mieux que SF v14).

| catégorie | part des positions | v5 | A | SF v14 | part du reste |
|---|---|---|---|---|---|
| ouverture et milieu (matériel ≥ 40) | 48 % | 39,1 | 33,7 | 35,2 | −29 % |
| milieu allégé (20-39) | 20 % | 89,8 | 78,1 | 72,6 | 43 % |
| finale (< 20) | 32 % | 68,9 | 59,7 | 52,9 | 86 % |
| matériel égal | 61 % | 32,4 | 28,4 | 28,4 | 1 % |
| écart de matériel 1-2 | 36 % | 94,2 | 81,4 | 75,6 | 82 % |
| écart de matériel ≥ 3 | 2 % | 192,5 | 158,9 | 140,9 | 17 % |
| aucun pion passé | 66 % | 35,4 | 30,9 | 31,5 | −16 % |
| pion passé | 24 % | 89,7 | 77,5 | 70,7 | 65 % |
| pion passé en 6e/7e | 10 % | 134,1 | 114,9 | 102,2 | 51 % |
| roi tranquille | 72 % | 42,5 | 37,0 | 37,0 | 1 % |
| roi exposé (3-5 attaques sur sa zone) | 24 % | 95,6 | 82,2 | 73,8 | 80 % |
| roi très exposé (≥ 6) | 4 % | 118,0 | 102,3 | 91,0 | 20 % |
| eval < 150 | 70 % | 26,0 | 23,0 | 24,9 | −55 % |
| eval 150-400 | 17 % | 72,4 | 66,7 | 56,7 | 66 % |
| eval ≥ 400 | 13 % | 217,7 | 180,5 | 163,2 | 89 % |
| ply < 20 | 22 % | 5,6 | 5,1 | 11,9 | −59 % |

(eval < 150 regroupe les deux lignes < 50 et 50-150 de `err_buckets.py`. Sortie complète,
y compris types de finales, tension et ply, dans `gsrv:Alcyon-serveur/erran/an.out`.)

Lecture :

- **Le réentraînement a gagné partout à peu près pareil** (−12 à −18 % d'erreur dans chaque
  catégorie) : ce n'est pas un type de position qui a fait les +120 Elo.
- **Sur les positions équilibrées, A est déjà au niveau de SF v14, voire meilleur** :
  matériel égal, roi tranquille, pas de pion passé, eval < 150, ouverture. Il y a là un biais :
  A est entraîné sur des données Leela de la même source que la validation, SF v14 sur
  d'autres données. Il ne faut pas en conclure que A évalue mieux ces positions en absolu.
- **Tout l'écart restant est dans les positions déséquilibrées** : finales (86 % du reste),
  écart de matériel (82 %), pions passés, roi exposé (80 %), avantage net ou décisif. Ces
  catégories se recouvrent, c'est le même groupe de positions vu sous plusieurs angles :
  celles où un camp a un avantage concret.
- L'écart d'erreur global entre A et SF v14 est petit (50,7 contre 48,2), alors que SF v14
  vaut encore environ 80 Elo de plus en partie. La recherche passe beaucoup de temps dans
  des positions déséquilibrées, plus rares dans ces données de parties Leela équilibrées :
  c'est là que la précision paie en Elo.

**Données à cibler** : des positions déséquilibrées et des finales, plutôt que d'autres mois
de test80 (même répartition que l'actuel). Candidats : `dfrc` (Fischer aléatoire, plus de
structures et de déséquilibres), les jeux de linrock riches en finales, ou des positions
tirées des arbres de recherche d'Alcyon (positions réellement visitées), rescorées par
une source forte.

## Test : données ciblées sur les positions déséquilibrées (25/11, aucun effet)

Fine-tuning de A, 10 époques, lr 6e-5 × 0,85^k (fin à 1,4e-5, recuit comparable à A),
lambda 1,0 (`tools/distill/run_mixdes.sh`). Mélange en positions : 74 % test80 juin 2024,
20 % de positions de test80 2023 juillet et octobre gardées par `filt_desequilibre.cpp`
(écart de matériel 1-4 points ou finale : 47 % des positions passent le filtre), 6 % de
`linrock/dual-nnue/hse-v1` (90 % d'écarts ≥ 5 points, 96 % d'evals ≥ 400). Données toutes
notées par Leela. Environ 1 h 25 de serveur.

Val loss : 0,00295 (A) → 0,00293. Analyse par type de position : **aucune catégorie ne bouge
de plus de 1 %** (matériel égal 28,4 → 28,5 ; écart 1-2 : 81,4 → 81,4 ; finales 59,7 → 59,8 ;
eval 150-400 : 66,7 → 66,3). Pas de dérive sur les positions équilibrées, pas de gain sur les
autres. Match contre A arrêté à 100 parties (−28 ± 44), inutile vu l'analyse.

Lecture : le réseau ne manquait pas de positions déséquilibrées (près de la moitié des
données d'origine en sont déjà). Passer de ~47 % à ~58 % n'apprend rien de nouveau. L'erreur
qui reste sur ces positions vient de la recette (fine-tunings successifs depuis v5), pas de la
répartition des données.

## Sonde à lr élevé (25/11, arrêtée)

A repris à lr 5e-4 (× 0,846 par époque, 22 époques prévues), données d'origine
(`tools/distill/run_hilr.sh`). La val loss monte à 0,00315 puis redescend à 0,00297 à
l'époque 9 (lr 1,1e-4), le même niveau que la lignée de A à lr comparable (0,00296 au début du
recuit de A). Arrêtée là : l'héritage de v5 ne semble pas être ce qui bloque, et le recuit
seul ne fait presque rien baisser dans cette lignée (0,00296 → 0,00295 pour A).

## Affinage sur les scores Leela BT4 (25/11)

Lu dans la recette SF (`recette-sf.md`) : SF entraîne sur les mêmes parties test80, mais
réévaluées par Leela BT4. Affinage de A sur test80 2024 mars à juin **réévalués par BT4**
(`xushawn/test80-bt4-relabel`, versions `-histfix`, 45 Go), lr 3e-4 × 0,854 par époque,
20 époques, lambda 1,0 (`tools/distill/run_bt4.sh`). Validation : 300 000 positions de février
2024 BT4. Environ 2 h 15 de GPU.

Val loss (scores BT4) : 0,00285 à l'époque 0, 0,00278 à l'époque 5, puis plateau vers
0,00275. Analyse par type de position sur 973 269 positions de février, scores BT4 :

| | v5 | A | SF v14 | BT4 |
|---|---|---|---|---|
| ensemble | 59,5 | 51,3 | 49,3 | **50,2** |
| matériel égal | 32,4 | 28,5 | 28,7 | 27,9 |
| écart de matériel 1-2 | 96,9 | 83,1 | 78,5 | 81,3 |
| finales | 73,1 | 63,6 | 57,6 | 62,2 |
| eval < 50 | 19,9 | 17,9 | 19,8 | 16,5 |
| eval 150-400 | 74,8 | 69,1 | 59,3 | **72,5** |
| eval ≥ 400 | 214,0 | 172,8 | 157,6 | 166,5 |

Premier gain d'erreur réel (−2 % presque partout, la moitié de l'écart avec SF v14), sauf sur
les avantages nets (eval 150-400), qui se dégradent. **Match contre A : +9,1 ± 15,7** (800
parties, 215 V / 194 D / 391 N) : positif mais non significatif.

Diagnostic sur les avantages nets : rapport médian éval du réseau / score BT4 sur les positions
150-400 (échelle globale ajustée par réseau) : A 0,81, BT4 0,80, SF v14 0,90. Nos réseaux
compriment les avantages nets. Ce n'est pas un facteur d'échelle global (l'échelle optimale vaut
1,00 pour A et BT4) : c'est la forme de la courbe. Cause probable : la fonction de perte. Par
défaut nnue-pytorch convertit l'éval du réseau avec une sigmoïde d'échelle 340 et la cible avec
380 (décalages 270), exposant 2,5, sans asymétrie ; la recette SF utilise 295,7 / 380,0,
décalages 285,3 / 289,1, exposant 2,43, asymétrie 0,23, une autre pondération par nombre de
pièces et par ply, et saute les 18 premiers plies.

## Affinage avec la perte de la recette SF (25/11, négatif)

Réseau BT4 repris avec la perte complète de la recette SF (valeurs ci-dessus), mêmes données
BT4, 16 époques, lr 1,5e-4 × 0,858 (`tools/distill/run_sfloss.sh`, 1 h 40 de GPU).

- Erreur globale (scores BT4) : 50,5 contre 50,2 pour le réseau BT4 ; eval 150-400 : 72,6
  (inchangé).
- Rapport éval/cible sur 150-400 : **0,792** (BT4 : 0,80, SF v14 : 0,90). L'échelle globale
  optimale passe à 1,08 : le réseau sort des evals plus petites, la compression ne se corrige pas.
- **Match contre le réseau BT4 : −21,7 ± 15,1** (800 parties, 197 V / 247 D / 356 N). 19 pertes au
  temps, 16 du réseau BT4 : sans elles l'écart serait plus défavorable encore (environ −27).

L'hypothèse « la forme de la perte cause la compression » est réfutée. Lecture la plus probable :
la compression des avantages nets est le rétrécissement classique d'un prédicteur incertain
(quand il ne sait pas, il ramène vers 0), donc un symptôme du manque de précision et non sa
cause. SF v14 comprime moins parce qu'il est plus précis : environ 75 fois plus de positions,
dont des positions d'arbres de recherche réévaluées par BT4.

Pertes au temps : à 10+0.1 et concurrence 30 sur ce serveur, les matchs du 25/11 en ont 15 à 19
pour 800 parties (réparties : 8 / 7 dans BT4 contre A), contre 0 à concurrence 24 pendant le run
long. Pour les prochains matchs : concurrence 24 au plus.

## Échelle de l'éval de v7 (26/11, neutre)

L'analyse d'erreur ajuste une échelle par réseau : 0,82 pour v5, 1,00 pour v7. v7 évalue donc
environ 18 % plus bas que v5, pour qui toutes les marges en centipions (RFP, futilité, NMP,
delta pruning, aspiration) et les entrées de score de la gestion du temps ont été réglées.
Option `eval_scale` ajoutée (pourcents, build SPSA, 100 par défaut, appliquée au réseau complet
et à la tête PSQT). SPRT `eval_scale=122` contre v7 (10+0.1, UHO, concurrence 12, Mac) :
**+1,0 ± 14,0 sur 718 parties** (158 V / 156 D / 404 N), LLR −0,10, arrêté. Les marges de v5
tolèrent l'échelle de v7 ; l'échelle par défaut est conservée.
