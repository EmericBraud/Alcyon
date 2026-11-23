# Evaluation par paliers de profondeur : mesures prealables

Idee : un grand accumulateur "en poupees russes" ; plus on est loin de la
racine, plus on n'utilise / ne met a jour qu'un prefixe de l'accumulateur,
suivi d'un petit reseau propre au palier (et ses buckets). Precision du NNUE
pres de la racine, vitesse tactique au fond de l'arbre.

## Mesure 1 : ou part le temps (echantillonnage macOS `sample`, 15 s, v6.1)

Thread de recherche uniquement (124 k echantillons), position de milieu de
partie, Hash 64 :

| Poste | Part |
|---|---|
| `Eval::eval` (maj incrementale de l'accumulateur + l0, en ligne) | 44,3 % |
| couches denses (`get_result`) | 26,8 % |
| features de menaces (`threats`) | 11,1 % |
| copies memoire (accumulateurs) | 6,9 % |
| tout le reste (recherche, movegen, TT...) | 10,9 % |

**L'evaluation represente ~89 % du temps.** Le moteur est limite par son
reseau, pas par sa recherche : toute baisse du cout du reseau la ou il est le
plus appele se traduit presque entierement en vitesse.

## Mesure 2 : ou ont lieu les evaluations (build `-DALCYON_EVAL_STATS`)

Evaluations NNUE (eval statique de negamax + stand pat de la quiescence) par
distance a la racine, 8 positions UHO :

| ply | 0,3 s (prof. finale ~14, 7,0 M evals) | 3 s (prof. finale ~20, 109 M evals) |
|---|---|---|
| 0-3 | 6,9 % | 0,6 % |
| 4-7 | 36,1 % | 9,4 % |
| 8-11 | 42,8 % | 32,6 % |
| 12-15 | 13,2 % | 36,5 % |
| 16-19 | 0,9 % | 17,4 % |
| 20+ | 0,0 % | 3,6 % |

## Lecture

1. Les evaluations sont concentrees dans la seconde moitie de l'arbre
   (ply > profondeur de la racine / 2 : ~ 60 a 90 % des evals). Un petit
   reseau dans cette zone toucherait l'essentiel du temps.
2. **La repartition se deplace avec la duree de recherche** : le pic est vers
   le ply 8-11 a 0,3 s, 12-15 a 3 s. Un seuil fixe en ply (ex. "256 entrees a
   partir du ply 15") ne servirait presque a rien en partie rapide. Comme pour
   la gestion du temps, les paliers doivent etre **relatifs a la profondeur de
   la racine** (ply / profondeur d'iteration), pas absolus.
3. Ordre de grandeur, optimiste : si ~85 % des evals passaient par un palier
   ~3x moins cher (accumulateur et couches proportionnels a la largeur ; le
   calcul des indices de menaces, lui, ne baisse pas), le temps total
   passerait de 1 a ~0,5, soit ~x2 en noeuds par seconde. A condition que la
   perte de precision au fond de l'arbre ne mange pas ce gain.

## Suite

Il faut un reseau entraine pour l'usage par prefixes (une perte par prefixe,
entrainement "Matryoshka"), calibre sur la meme cible pour que les paliers
restent coherents (TT, marges d'elagage, correction history). Prototype
suggere : deux paliers (1024 et 256), palier choisi par ply / profondeur de
la racine.

## Pipeline d'entrainement (retrouve le 2026-11-23)

Rien n'etait versionne a part le sous-module ; la procedure se deduit du
moteur (`src/engine/eval/nnue/nnue_model.hpp`) :

- **nnue-pytorch au commit `4289208fe20cc6ec8753e5ee14c2f210de783ff0`** (le
  sous-module pointait sur `9f72946`, plus recent, dont les signatures
  d'architecture ne correspondent plus a v5) ;
- features **`Full_Threats+HalfKAv2_hm^`** (60 720 + 22 528 = 83 248),
  hyperparametres par defaut : L1 = 1024, L2 = 32, L3 = 32, 8 buckets PSQT,
  8 piles de couches ;
- export sans compression : `--ft-compression none` (le moteur ne lit pas le
  leb128).

Verification sur le serveur GPU : `serialize.py v5.nnue v5.pt` puis
`serialize.py v5.pt v5_rt.nnue --ft-compression none --description <meme
texte>` redonne **un fichier identique octet pour octet**, et le moteur fait le
meme bench (16763) avec le reseau reexporte. On peut donc fine-tuner v5.

Serveur : `tools/gpu/setup_gpu.sh` (RTX 5090, PyTorch cu128, g++-14 pour le
moteur en C++23, binpacks `linrock/test80-2024` janvier et fevrier 2024,
`.min-v2.v6`).
