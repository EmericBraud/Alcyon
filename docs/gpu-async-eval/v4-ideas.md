# Idées pour v4 — pistes pour dépasser la NNUE en précision

Statut : notes de brainstorming, rien n'est implémenté. La v3 (plans
d'attaque décomposés par type+couleur, 14 blocs × 160 canaux, ~6,9M
paramètres) continue son entraînement en parallèle — voir
`experiment-v1.md` pour le contexte général et les résultats de la
v1. Ce document liste les pistes envisagées pour une itération
suivante, avec leur statut de décision.

## Contexte : pourquoi la NNUE reste devant

La NNUE de chess26 n'est pas "petite" malgré l'inférence rapide :
sa table de features (`HalfKAv2_hm`, 22 528 features → 1024 neurones)
pèse à elle seule ~23M de paramètres, plus ~0,5M pour les 8 buckets
de layer-stack et de PSQT — soit **~23,5M au total, contre ~6,9M**
pour le CNN v3, un facteur ~3,4×. Elle bénéficie aussi de features
d'entrée conçues sur-mesure pour les échecs (position relative au roi
encodée directement dans l'index de feature) et d'un pipeline
d'entraînement mûri depuis des années par la communauté Stockfish
sur des milliards de positions — pas juste "plus de paramètres".

## Pistes retenues (priorité décroissante)

### 1. Scaler encore le réseau — retenue

Passer à quelque chose comme 20 blocs × 224-256 canaux (échelle
Leela Chess Zero) pour se rapprocher de la capacité brute de la
NNUE (~23,5M paramètres). Coût : temps d'entraînement plus long,
atténué par les optimisations bf16/`torch.compile`/`cudnn.benchmark`
déjà en place (gain mesuré ×2,4 en débit sur la v3).

### 2. Plans d'entrée relatifs au roi — retenue

Ne pas répliquer le bucketing par roi complet de la NNUE (32
buckets) — inadapté à un CNN, qui voit déjà la position du roi
nativement via les convolutions spatiales (contrairement à la NNUE,
réseau non-spatial qui a besoin de ce mécanisme pour compenser).
À la place : ajouter 1-2 plans "distance de Chebyshev au roi
(nôtre/adverse)" pour donner un raccourci direct vers la notion de
zone de sécurité du roi, sans fragmenter l'entraînement par position
de roi.

### 3. Granularité de phase alignée sur la NNUE — retenue

Passer de 4 à 8 buckets de phase (`PHASE_BUCKET_BOUNDARIES`), pour
matcher la granularité des 8 buckets de layer-stack de la NNUE.
Changement mécanique, faible risque avec le volume de données
disponible (dataset de ~9,2 Go).

### 4. Augmentation par symétrie miroir gauche-droite — écartée

**Décision : écartée**, sur objection justifiée de l'utilisateur.
L'idée initiale (flip des fichiers a↔h pour doubler gratuitement le
volume de données vues) est en fait **cassée par construction** :
les fichiers ne sont pas symétriques autour de la position de départ
du roi (e1 ↔ d1, pas e1 ↔ e1), donc un flip naïf déplacerait le roi
et inverserait roque côté roi / côté dame sans relabelliser les
droits au roque en conséquence (`us_castle_kingside` ↔
`us_castle_queenside`). Faisable en théorie avec la relabellisation
correcte, mais ce n'est plus un gain "gratuit" — risque réel de
corruption silencieuse des données si mal implémenté, pour un
bénéfice incertain. Non prioritaire.

### 5. Entraînement plus long — retenue, en cours

Le plateau de la v1 à 180-200k steps coïncidait avec la fin de la
décroissance cosine du LR (lr=0 à step 200 000), pas nécessairement
avec un maximum réel de capacité du modèle. Décision : **la v3 n'est
pas abandonnée** — son run en cours est repris avec l'horizon étendu
à **300 000 steps** (au lieu de 100 000), via `--resume-from` +
`--steps 300000` (pas `--extend-steps`, puisque le run original
n'était pas terminé — la décroissance cosine se réajuste simplement
sur le nouvel horizon).

Corollaire pour une v4 plus grosse (piste #1) : un réseau plus
grand a besoin de *plus* de steps pour converger, pas moins — si le
réseau est scalé, le schedule devra probablement être allongé en
proportion.

### 6. Curriculum / pondération des positions difficiles — non retenue pour l'instant

Sur-échantillonner les positions tactiquement chargées (où l'écart
au static eval est le plus dur à prédire) plutôt qu'un tirage
uniforme. Pas priorisée faute d'un moyen simple de mesurer la
"difficulté" d'une position avant l'entraînement sans un premier
passage complet — à reconsidérer si les pistes 1-3-5 ne suffisent
pas.

### 7. Tête auxiliaire de matériel — retenue, priorité basse

Ajouter une petite tête secondaire (utilisée uniquement pendant
l'entraînement, jamais à l'inférence) qui prédit le différentiel
matériel de la position à partir des features intermédiaires du
trunk, avec son propre terme de loss (MSE) additionné à la loss
principale. Cible triviale à calculer (comptage de pièces par type,
pas de données externes nécessaires).

Intuition : le matériel est le signal le plus simple et le moins
ambigu aux échecs — forcer le réseau à le représenter explicitement
tôt dans le trunk peut accélérer/stabiliser l'apprentissage de
nuances positionnelles plus subtiles (technique classique de
supervision auxiliaire en deep learning multi-tâche). Risque : le
PSQT linéaire actuel capture déjà une partie de ce signal, donc le
gain marginal est incertain — piste la moins prioritaire des trois
retenues (#1, #2, #3), à tester en dernier.

## Ordre d'implémentation suggéré pour v4

1. Scaling du réseau (#1) + granularité de phase (#3) — changements
   mécaniques, faible risque, à combiner dans le même changement
   d'architecture.
2. Plans relatifs au roi (#2) — nouveau signal d'entrée, à valider
   isolément (ablation) avant de le considérer acquis.
3. Tête auxiliaire de matériel (#7) — expérimentale, en dernier.

Le schedule d'entraînement (#5) s'applique quel que soit
l'architecture retenue : prévoir un horizon de steps proportionnel à
la taille du réseau dès le lancement, plutôt que d'enchaîner des
extensions a posteriori.
