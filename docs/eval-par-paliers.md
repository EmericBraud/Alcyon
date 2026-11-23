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

## Mesure 3 : precision d'un palier reduit (`tools/gpu/tiers_probe.py`)

v5 fige ; petite tete (memes L2/L3 et 8 buckets) entrainee 3000 pas
(batch 16 384) par distillation de la sortie de v5, sur k neurones par
perspective (2k entrees d'accumulateur par perspective sur 1024).
Validation : 655 k positions de fevrier 2024. Ecart a la sortie de v5 en MSE
sur la probabilite de gain (sigmoide de cp / 400). Etalon : ecart de v5 lui-meme
aux donnees, 0,0085.

| entrees / perspective | neurones par importance | prefixe brut |
|---|---|---|
| 128 / 1024 | 0,0097 | 0,0130 |
| **256 / 1024** | **0,0068** | 0,0093 |
| **512 / 1024** | **0,0036** | 0,0050 |

- A 256 entrees, le palier reduit est plus proche de v5 que v5 ne l'est des
  donnees : l'erreur ajoutee est inferieure a l'erreur deja presente.
- Choisir les neurones par importance (activation moyenne x poids sortants)
  enleve ~30 % d'erreur : il faudra permuter le FT avant de prendre un prefixe.
- Borne superieure de l'erreur : le FT est fige. Un fine-tuning "Matryoshka"
  (le FT apprend a concentrer l'information en tete) devrait faire mieux.
- L'ecart moyen en cp (110 a 235) est domine par les positions deja decidees
  (evals de milliers de cp) et ne doit pas etre lu tel quel.

## Integration : quels noeuds en precision complete ?

Contrainte : l'accumulateur est incremental. Passer de complet a reduit en
descendant est gratuit ; remonter a complet sous un noeud reduit demande de
rattraper les entrees manquantes (remonter au dernier ancetre complet, ou
recalcul depuis les features). La regle doit donc etre surtout monotone.

Plan propose :
1. complet sur la PV, le coup TT et les extensions, et a faible ply relatif ;
   reduit dans les sous-arbres reduits par LMR et en quiescence ;
2. evaluation en deux temps : palier reduit d'abord, rattrapage et eval
   complete seulement si le resultat tombe pres de alpha / beta ;
3. la petite tete predit aussi sa propre erreur (une sortie de plus, apprise
   sur l'ecart de distillation) : rattrapage si erreur prevue forte ET proche
   d'une borne.

Chaque etape : d'abord en noeuds par seconde, puis en parties.

## Integration moteur (branche `eval-tiers`)

### Reseau

- `tools/gpu/tiers_export.py` : ordre d'importance des 512 neurones
  (activation moyenne x poids sortants), petite tete (k = 128 neurones, soit
  256 entrees d'accumulateur sur 1024 par perspective) entrainee 8000 pas par
  distillation de v5 fige : **MSE proba 0,0064 contre v5** ; permutation
  enregistree (`ft.perm`) et `serialize.py --ft-perm` donne **v5 permute
  (`v5p.nnue`), de sortie identique** (bench 16763). Le mode paliers n'a de
  sens qu'avec ce reseau permute.
- Deux exports de la tete : flottante (`tier_head_weights.hpp`) et
  **quantifiee comme la grande** (`tier_head_q.hpp`, `quantize_fc_layer` de
  nnue-pytorch, int8 / int32 ; 3 a 5 poids par bucket rognes dans la plage
  int8, la tete n'ayant pas ete rognee pendant l'entrainement).
- Garde-fous ajoutes apres deux pertes de temps : poids de la tete sauves
  juste apres l'entrainement, `--load-head` + `--perm` pour refaire l'export
  seul (une tete rechargee DOIT reprendre sa permutation : l'ordre recalcule
  depend des lots tires et differe), essai de bout en bout a 20 pas avant un
  long entrainement.

### Moteur

- `AccumulatorLayer::update_feature_narrow` : mise a jour des seules tranches
  [0, K) et [512, 512 + K) (le produit par paires lit i et i + 512).
- `NnueEval::materialize_narrow` : rattrapage partiel ; `narrow_from` retient
  la premiere profondeur appliquee en partiel. Une eval complete demandee sous
  un etat partiel revient avant lui (les instantanes restaurent l'etat
  complet) et rejoue en complet. Mesure : 2 419 replis pour 686 000 evals
  reduites, negligeable.
- Choix du palier : hors PV (beta - alpha == 1) et
  `ply * 100 >= tier_ply_pct * profondeur de l'iteration` (negamax et
  quiescence). `tier_mode` 0 / 1 (tete flottante) / 2 (tete quantifiee).

### Vitesse (Mac, 1 thread, 8 positions UHO x 2 s, v5 permute)

| mode | noeuds/s | gain |
|---|---|---|
| sans paliers | 692 000 | - |
| tete flottante, 1re version | 728 000 | +2 % |
| tete flottante, couches en axpy | 794 000 | +15 % |
| **tete quantifiee (int8, SIMD)** | **1 124 000** | **+62 %** |

Profil de la 1re version flottante : la petite tete prenait 43 % du temps
(produits scalaires flottants non vectorises sans reassociation), alors que
la mise a jour partielle de l'accumulateur etait tombee a 13 % (contre 44 %
pour la mise a jour complete). La quantification a regle le probleme.

## Premier match : -200 Elo, et pourquoi

Matchs rapides sur le serveur GPU (128 coeurs, 10+0.1, Hash 16, v5 permute
des deux cotes, tete quantifiee k = 128) :

| seuil (tier_ply_pct) | Elo | parties |
|---|---|---|
| 50 % | -196 +- 37 | 295 (arrete) |
| 70 % | -175 +- 39 | 274 (arrete) |

Diagnostic (commande UCI `evaltier` ajoutee : eval complete et reduite de la
position) : les tetes flottante et quantifiee concordent a quelques cp pres,
l'integration est donc juste ; mais la petite tete s'ecarte de la complete
de 30 a 90 cp sur des positions ordinaires. Mesure propre ensuite
(`tiers_matryoshka.py`, positions |eval| < 600 cp) : **RMS 145 cp, MAE 97 cp**.

**Erreur d'interpretation de ma part** : j'avais compare l'ecart petite tete /
v5 (MSE 0,0064 en probabilite) a l'ecart v5 / donnees (0,0085) et conclu que
le palier ajoutait moins d'erreur que v5 n'en faisait deja. Mais l'ecart aux
donnees est gonfle par le bruit de la cible : ce n'etait pas un etalon. Autour
d'une eval nulle, 0,0064 en probabilite correspond a ~100 cp. Il faut juger
un palier en centipions sur les positions disputees.

Conclusion : une petite tete apprise sur un accumulateur fige est trop
imprecise. D'ou le fine-tuning Matryoshka (`tools/gpu/tiers_matryoshka.py`) :
tout le reseau est reentraine depuis v5 (Adam, lr 3e-5 reseau / 3e-4 tete,
cosinus), perte = perte complete (cible des donnees, lambda 0,6 comme v5) +
perte de la petite tete (meme cible) + 20 x distillation vers la complete ;
suivi de la perte complete (ne doit pas se degrader) et de l'ecart en cp.
