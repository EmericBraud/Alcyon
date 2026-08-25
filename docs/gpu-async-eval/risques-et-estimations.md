# Risques et estimations Elo

## Pourquoi pas une intégration bloquante ou directe dans l'arbre

### Latence vs débit de nœuds

Un NNUE CPU évalue en dizaines/centaines de nanosecondes. Même un GPU
rapide, en appel synchronisé, tourne en dizaines de microsecondes
minimum par requête isolée (hors gros batch) — 100 à 1000x plus lent.
Bloquer le thread de recherche pour attendre un résultat GPU coûterait
largement plus en débit de nœuds explorés que le gain de précision
apporté.

### Problème de causalité

On ne sait qu'un nœud est "sur la PV" qu'après l'avoir exploré,
potentiellement après que sa valeur a déjà influencé des coupures
alpha-beta ailleurs dans l'arbre. Un résultat GPU qui arrive après
coup ne peut pas être injecté rétroactivement dans une décision déjà
prise sans violer l'invariant de cohérence temporelle de l'alpha-beta.
Aggravé en multithread (Lazy SMP) : un score écrit "en retard" dans la
TT partagée peut être lu par un autre thread dans un contexte
(profondeur, fenêtre alpha-beta) différent de celui où il a été
calculé.

### Solution retenue : fire-and-forget + lecture non-bloquante

Le design retenu (voir [architecture.md](architecture.md)) résout le
problème de latence (jamais d'attente) mais pas totalement la
causalité : le résultat GPU, quand il arrive, est nécessairement en
retard sur le nœud qui l'a déclenché. La valeur n'est exploitée que
lors d'une **revisite ultérieure** de la même position (transposition,
itération ID suivante) — jamais rétroactivement sur le nœud d'origine.
C'est le même principe qu'une TT classique.

## Normalisation des scores

Normaliser le score CNN sur la même échelle que le NNUE (centipawns,
même conversion win-prob) est nécessaire mais pas suffisant : ça
résout l'incohérence de magnitude, pas le problème de latence/
causalité ci-dessus.

## Risque thermique / partage de puissance (Apple Silicon)

Sur un SoC, CPU et GPU partagent le même budget thermique et parfois
la même limite de puissance. Un CNN tournant en continu sur le GPU
pendant une recherche multithreadée intensive sur CPU (Lazy SMP) peut
faire baisser la fréquence turbo du CPU par partage thermique, ce qui
annulerait une partie du gain. À mesurer empiriquement (nps CPU avec
et sans charge GPU soutenue) avant de conclure sur le gain net.

## Non-déterminisme

La disponibilité du résultat GPU dépend du timing (charge GPU, ordre
d'arrivée des requêtes) : deux exécutions de la même recherche peuvent
produire des arbres différents. Acceptable en jeu (gain net attendu en
moyenne), mais complique les tests SPRT — soit désactiver le GPU en
mode test déterministe, soit accepter plus de bruit dans les résultats
et compenser avec davantage de parties.

## Estimation du gain Elo

Le design retenu ne touche qu'une **petite fraction** des nœuds : ceux
qui sont à la fois (a) sur la PV ou stables (EXACT + depth élevée),
(b) revisités par transposition à l'itération suivante, et (c) pour
lesquels le calcul GPU a eu le temps de finir.

- **Cas optimiste réaliste : 15-40 Elo.** Ordre de grandeur comparable
  à une correction history (correction ponctuelle sur des positions
  déjà vues, pas remplacement global de l'éval).
- **Cas très optimiste (le CNN corrige régulièrement des erreurs qui
  changent le coup choisi à la racine) : jusqu'à 50-60 Elo.** Dépend
  fortement du taux de désaccord NNUE/CNN sur des positions qui
  comptent réellement pour la décision finale.
- **Hors de portée avec ce design : le gain d'un remplacement complet
  de l'éval à chaque nœud** (façon upgrade vers un NNUE plus gros,
  potentiellement 50-150 Elo selon l'écart de taille) — écarté
  précisément pour des raisons de débit/latence.

## Comparaison coût/bénéfice avec les alternatives déjà identifiées

En pur ratio effort/Elo, ce plan est probablement moins rentable que :
- l'ajout d'un flag "improving" dans `negamax.cpp` (15-30 Elo, faible
  risque, faible effort),
- l'history pruning sur les coups quiets (10-20 Elo),
- un plus gros réseau NNUE utilisé à 100% de l'arbre (potentiellement
  bien supérieur, car il bénéficie à tous les nœuds et non à une
  fraction).

Ce plan GPU async reste pertinent comme piste d'exploration
architecturale ou si la taille de NNUE praticable en CPU est déjà
maximisée, mais ne doit pas être vu comme le levier Elo prioritaire.
