# Plan : passer du canal « valeur » au canal « ordonnancement »

Ce document décrit la direction décidée après la revue de
[consultative-eval-measurements.md](consultative-eval-measurements.md), et
surtout **l'ordre dans lequel on va essayer de la falsifier**. Il n'est pas
un compte rendu : les mesures se consignent dans l'autre fichier.

## 1. Pourquoi on change de canal

Trois chiffres mesurés ferment la direction actuelle.

**Le rapport de débit est de 1 à 256.** La recherche parcourt ~4 M nœuds/s
(10 threads) ; l'ANE évalue 15 600 positions/s. Même avec un ciblage
parfait, on raffinerait 0,4 % des feuilles.

**On est très loin de ce plafond.** Sur une recherche de 10 s : ~39 M nœuds,
80 000 positions calculées, 6 300 relues (7,9 %), dont ~55 % changent une
décision de stand-pat. Soit **un nœud sur 12 000 touché, un sur 22 000 qui
change**. C'est l'empreinte totale du sous-système sur la recherche, et elle
explique l'Elo à 0 ± bruit sans avoir à invoquer quoi que ce soit d'autre.

**Le ciblage ne peut pas être réparé.** Un évaluateur asynchrone doit avoir
son score prêt *avant* que la recherche n'atteigne le nœud, donc il faut
prédire quels nœuds seront visités. Or l'ensemble des nœuds visités par
alpha-beta est déterminé par la recherche elle-même : il dépend des coupures,
qui dépendent des scores trouvés en route. On prédit la sortie d'un processus
dont l'entrée est sa propre sortie. C'est pourquoi toutes nos heuristiques
plafonnent vers 8 %, et pourquoi passer le vrai `tt_move`/`ply`/`prev_move`
au tri des candidats (commit `63f44a8`) n'a rien donné.

## 2. Les deux règles qui contraignent tout le reste

**Règle de comparabilité.** Le CNN ne peut intervenir qu'à quantité
comparable : statique contre statique, ou recherche contre recherche. Il n'a
le droit de remplacer une valeur que là où il n'existe aucun résultat de
recherche. C'est ce qui disqualifie le départage des coups racine que
j'avais d'abord proposé : opposer une éval à 0 pli à un verdict alpha-beta à
20 plis, c'est jeter l'information qui fait la force du moteur.

**Asymétrie du risque.** Une valeur fausse corrompt le résultat. Un
ordonnancement faux ne coûte que du temps — alpha-beta cherche toujours
tout, simplement dans un moins bon ordre. Un hint n'est pas une prétention
sur la valeur, c'est une suggestion.

Croisées, ces deux règles ne laissent que deux canaux : les valeurs de
feuille (légitime mais plafonné à 1:256) et **l'ordonnancement** (non
plafonné, et le seul où se tromper est bon marché).

## 3. Ce qu'on jette aujourd'hui

Chaque tâche soumise évalue les 4 meilleurs enfants d'un parent. On garde les
4 **valeurs**, stockées comme évals de feuille — le canal plafonné et
risqué — et on jette leur **ordre relatif**, qui est le canal non plafonné et
sans risque. C'est à l'envers sur les deux critères, et le travail de calcul
est déjà fait.

## 4. Phase 0 — falsifier avant d'écrire du code

Rien ne justifie de toucher au `MovePicker` avant de savoir si le CNN a
quelque chose à dire sur l'ordre. Le test ne demande aucun changement de
comportement.

Pour chaque parent soumis, le thread GPU connaît après inférence le coup que
le CNN préfère parmi les candidats. On le compare au coup que la recherche a
finalement retenu à ce nœud (relu dans la TT principale, plus tard dans la
recherche). Trois taux à mesurer :

- **accord CNN** : le coup préféré du CNN est-il celui que la recherche a
  retenu ?
- **accord de référence** : le premier coup de notre tri actuel l'est-il ?
- **accord du coup TT** seul, comme borne haute triviale.

**Condition d'arrêt : si l'accord CNN ne dépasse pas l'accord de référence,
on s'arrête là et on écrit le résultat négatif dans le fichier de mesures.**
Coût : deux compteurs et une relecture de TT sur le thread GPU. Aucun risque
pour la recherche.

Attention au piège de méthode : ne comparer que sur les nœuds où la
recherche a *réellement* eu le choix (au moins deux coups explorés), sinon
l'accord est gonflé par les nœuds à coup unique.

## 5. Phase 1 — le design, si la phase 0 passe

**Stockage.** L'entrée de `gpu_tt` n'utilise que les bits 0-31 (score 16,
depth 8, age 8) ; les bits 32-63 sont libres. Un coup compressé y tient
(from 6 bits + to 6 bits + promo 2 bits), plus un bit de validité. Pas un
octet de mémoire en plus, pas de seconde table.

Deux chemins d'écriture distincts, tous deux sur le thread GPU :
`store_score(clé_enfant, ...)` comme aujourd'hui, et
`store_hint(clé_parent, coup)`. Comme ce thread est l'unique écrivain, le
read-modify-write est sans course (`store()` en dépend déjà pour compter les
`redundant_stores`) : chaque chemin **préserve** le champ de l'autre, donc
une position qui est à la fois parent soumis et enfant d'une autre
soumission garde ses deux informations.

**Consommation.** Une sonde par **nœud**, pas par coup : le `MovePicker` est
construit par nœud, il sonde `gpu_tt` une fois avec le hash du nœud courant
et accorde un bonus au coup indiqué. C'est ce qui rend le design abordable —
la variante « sonder la clé de chaque enfant au moment du tri » a été
écartée parce qu'il n'existe pas d'aide « hash après coup » dans
`zobrist.hpp` : il faudrait soit un `play`/`unplay` par coup trié, soit
réécrire le zobrist incrémental (castling, en passant, promotions), où un
bug sonde silencieusement la mauvaise entrée.

**Magnitude du bonus.** Sous le coup TT (9600) et sous les killers (8000) :
le hint est « un candidat calme sérieux », pas « plus fiable qu'une coupure
prouvée ». Départ proposé vers 6000, au-dessus de l'history et en dessous
des counter-moves (7500). À régler, pas à deviner.

**Portée.** Limitée aux nœuds proches de la racine, où les sous-arbres sont
gros et où un bon premier coup se rembourse. Une porte de profondeur, comme
pour les soumissions.

## 6. Phase 2 — mesurer sans passer par l'Elo

La métrique principale n'est pas l'Elo mais le **nombre de nœuds pour
atteindre une profondeur fixe**, sur une suite de positions. Un meilleur
ordonnancement élague davantage, donc coûte moins de nœuds : c'est objectif,
et surtout la variance est bien plus faible que celle d'un match. Les
mesures NPS de cette session ont montré 74 % d'écart entre deux runs
identiques sur batterie ; un comptage de nœuds à profondeur fixe est
déterministe à thread unique.

Protocole : `Threads 1` pour supprimer le non-déterminisme du SMP, `go depth
N` sur une dizaine de positions, avec et sans le hint, et comparer les
totaux. Complément utile : le taux « le premier coup essayé était le
meilleur » aux nœuds à choix multiple.

L'Elo ne vient qu'après, **sur secteur** : voir l'avertissement dans
`gpu_config.hpp` sur `transposition_submit_min_depth`.

## 7. Plafonds connus d'avance

À garder en tête pour ne pas surinterpréter un résultat médiocre.

L'ordonnancement du moteur est **déjà bon** : coup TT, captures triées par
SEE, killers, counter-moves, history, continuation history. Le hint doit
battre cet empilement, ce qui n'est pas acquis.

Notre modèle n'a **pas de tête policy**, seulement une tête value (l'étude
des architectures cibles est dans
[ane-architecture-study.md](ane-architecture-study.md)). Ordonner
en évaluant les enfants un par un est une approximation coûteuse de ce qu'une
tête policy donnerait directement. C'est une limite du modèle, pas de
l'intégration — et si la phase 0 est proche du seuil, entraîner une tête
policy serait le vrai levier plutôt que de raffiner l'intégration.

Le CNN est corrélé à **0,91** avec NNUE (pente ~1, offset corrigé). Il
n'apporte que les 17 % de variance résiduelle, de signe inconnu. Ça borne ce
canal comme ça bornait l'autre ; la différence est que l'ordonnancement n'est
pas en plus divisé par 256.

## 8. Ce que cette direction ne résout pas

Elle ne rend pas le sous-système utile en soi : elle déplace le pari sur un
canal où le plafond arithmétique et le risque sont tous deux meilleurs. Si la
phase 0 échoue, la conclusion honnête est que **ce modèle-ci** n'a rien à
apporter à **cette recherche-ci**, et que la suite est côté entraînement
(tête policy, ou réseau nettement plus petit pour changer le rapport 1:256),
pas côté intégration.

---

## Résultat de la phase 0 : NÉGATIF, et la direction s'arrête

Mesuré sur 5 positions, **51 821 échantillons** :

| prédicteur | accord avec le coup que la recherche a finalement retenu |
|---|---|
| préférence du CNN | **18,75 %** |
| heuristiques d'ordonnancement | **24,66 %** |
| écart | **−5,91 points** |

L'heuristique gagne sur **5 positions sur 5**. Par position, l'écart va de
−1,1 à −30,5 points, jamais en faveur du CNN.

La conclusion est donc plus forte qu'un simple « pas de gain » : le signal
d'ordonnancement tiré du CNN est **substantiellement moins bon** que
l'empilement déjà en place (MVV-LVA/SEE, killers, counter-moves, history,
continuation history). La condition d'arrêt de la section 4 est remplie.

### Le premier test était faux, et c'est instructif

La première version mesurait CNN 56,1 % contre heuristiques 54,1 % sur 1 948
échantillons — un écart de +2,0 points, ambigu. Elle était **fausse sur les
deux axes à la fois** :

- **Mauvaise population.** Elle n'échantillonnait que les nœuds ayant *déjà*
  un coup TT. Or à ces nœuds le coup TT reçoit un bonus de 9600 et passe
  premier quoi que dise le CNN : un hint n'y sert à rien par construction.
- **Mauvaise cible.** Elle prenait ce même coup TT comme cible, donc mesurait
  la capacité du CNN à prédire une information que le moteur possédait déjà.

La version correcte ne retient que les nœuds soumis **sans** coup TT — la
seule population où un hint servirait — et compare, plus tard, au coup que la
recherche a effectivement conclu sur ce nœud. D'où la table de hints différée
(`GpuTT::store_ordering_hint`, délibérément séparée du cache de scores : y
ranger un hint obligerait à écrire un score bidon, relu ensuite comme un vrai
stand-pat par qsearch).

Les deux taux s'effondrent au passage (56 %/54 % → 19 %/25 %), ce qui est
cohérent : prédire le verdict d'une recherche qui n'a pas encore eu lieu est
bien plus dur que « prédire » un coup déjà présent dans la TT. Et le
handicap que j'imposais à l'heuristique (l'aveugler au coup TT) devient sans
objet ici, puisqu'il n'y avait pas de coup TT à masquer — la comparaison est
donc franche, contre l'ordonnancement réellement déployé.

### Ce que ça ne démontre pas

Le prédicteur testé est la tête **value**, lue par argmin des évals d'enfants.
Ce n'est pas une tête **policy** entraînée, qui lirait le parent directement
et serait entraînée sur exactement cette cible. Le proxy sous-estime donc ce
qu'une policy pourrait faire.

Mais l'écart à combler n'est plus de 2 points d'ambiguïté : c'est **5,9 points
dans le mauvais sens**, sur 51 821 échantillons et 5 positions sur 5. Avant
d'investir dans une tête policy (entraînement absent du dépôt, dataset à
générer, réseau à concevoir), il faudrait une raison de croire qu'elle
franchirait cet écart. Le poids de la preuve a changé de camp.

### Bilan des deux canaux

- **Valeurs** : plafonné à 1 feuille sur 256 par le rapport de débit,
  corrélation 0,91 avec NNUE, un nœud touché sur 12 000. Elo mesuré
  −2,5 ± 27 sur 139 parties.
- **Ordonnancement** : −5,91 points contre l'ordonnancement existant.

Les deux canaux sont mesurés, aucun ne paie. L'effort rapporterait davantage
sur la recherche ou le NNUE.

### Correction : c'est la MÉTHODE qui échoue, pas le CNN

La conclusion ci-dessus sur-interprétait. Contrôle méthodologique : refaire
exactement la même mesure en remplaçant le CNN par **NNUE**, évaluateur bien
plus précis, via le même `argmin` des évals d'enfants.

Sur 15 251 échantillons, 3 positions :

| prédicteur | accord avec le coup retenu par la recherche |
|---|---|
| CNN | 20,22 % |
| **NNUE (même méthode)** | **26,59 %** |
| heuristiques d'ordonnancement | **39,13 %** |

Un évaluateur nettement meilleur n'achète que **+6,4 points**, et perd encore
**12,5 points** contre les heuristiques.

Donc le facteur limitant n'est pas la qualité du modèle : c'est la méthode.
Différencier les évals statiques de positions qui ne divergent que d'un pli
est une mauvaise façon de prédire le verdict d'une recherche profonde, quel
que soit l'évaluateur. Les écarts entre frères font 10-50 cp, l'erreur
absolue de l'éval est du même ordre — le bruit domine le signal.

**La phase 0 ne peut donc pas conclure sur une tête policy entraînée.** Ce
qu'elle établit est plus étroit, et reste utile : on ne peut pas obtenir de
l'ordonnancement gratuitement en relisant une tête *value*. La condition
d'arrêt de la section 4 était mal spécifiée — elle confondait « le signal
qu'on peut extraire aujourd'hui » et « ce que le modèle pourrait apprendre ».

Lecture constructive, d'ailleurs : les heuristiques gagnent en utilisant des
features **du coup** (SEE, killers, counter-moves, history), pas l'éval de la
position résultante. Une tête policy est exactement une version apprise de
ça. L'évidence pointe donc *vers* une policy comme bon instrument, et non
contre.

### Le prochain verrou, avant d'investir dans l'entraînement

Reste à borner le gain accessible, et ça se mesure sans rien entraîner :
compter les nœuds à profondeur fixe (`Threads 1`, déterministe) avec
l'ordonnancement actuel, puis avec le meilleur coup placé d'office en tête
(oracle). L'écart est le **maximum** que n'importe quelle policy pourrait
rapporter. Si l'oracle n'économise que quelques pourcents, aucune policy ne
vaut l'investissement ; s'il en économise 30 %, la cible est claire et on
connaît d'avance la barre à franchir (39,13 % de top-1).
