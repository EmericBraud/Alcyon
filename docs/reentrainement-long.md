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
