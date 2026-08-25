# Architecture

## Vue d'ensemble du pipeline

1. Le CPU explore normalement l'arbre avec le NNUE (aucun changement
   du hot path de `negamax.cpp`).
2. Un nœud à la profondeur N où le NNUE est appelé pour l'éval
   statique est une **feuille** pour cette branche : la recherche ne
   génère aucun coup au-delà. Le thread de recherche fait donc la
   capture la moins chère possible — copier l'état brut de la
   position (bitboards, structure POD, memcpy) — pour les positions
   qui nous intéressent :
   - nœuds sur la PV actuelle,
   - nœuds avec bound `EXACT` et depth élevée dans la TT.
3. Ces positions brutes sont poussées (non-bloquant) dans un ring
   buffer préalloué vers un **unique cœur CPU dédié** à la
   préparation GPU (un seul E-core sur Apple Silicon, distinct des
   P-cores utilisés par les threads de recherche Lazy SMP).
4. Ce cœur de préparation fait tout le travail que le thread de
   recherche n'a pas fait, puisqu'aucun move ordering n'existe à un
   nœud feuille :
   - `generate_moves()` complet sur la position reçue,
   - un tri/filtre léger pour retenir les K coups à explorer à N+1
     (captures, échecs, ou un ordre simplifié type MVV-LVA — pas
     besoin d'être aussi élaboré que le move picker de la recherche),
   - `make_move()` pour chacun des K coups retenus → positions à N+1,
   - encodage des features de chaque position enfant.
5. Le cœur de préparation batch les features encodées et lance
   l'inférence CNN via Metal/MPS (mémoire unifiée Apple Silicon → pas
   de copie CPU/GPU explicite, juste un buffer partagé).
6. Dès qu'un résultat est prêt, il est écrit dans un cache séparé :
   `zobrist_key -> {cnn_score, timestamp}`.
7. À l'itération suivante (ID depth N+1), quand le CPU (re)visite une
   position, le `probe()` de la TT principale consulte aussi ce cache
   en lecture seule (lookup non-bloquant, coût quasi nul).
8. Si une entrée existe et n'est pas trop ancienne, le score CNN sert
   à ajuster l'éval statique du nœud (ex. moyenne pondérée avec le
   NNUE, ou override si désaccord au-delà d'un seuil) — jamais à
   réécrire rétroactivement une décision de pruning déjà prise.

Le coût de préparation (movegen + tri + make_move + encodage) reste
isolé sur cet unique cœur dédié : il ne rentre jamais en compétition
avec les threads de recherche pour les cycles CPU, même s'il est plus
élevé qu'un simple encodage de features (puisqu'il inclut le movegen
qu'un nœud feuille n'aurait sinon jamais déclenché).

## Pourquoi ce timing (fin d'exploration de profondeur N, pas fin d'ID)

Le facteur de branchement de l'arbre fait que le délai entre la
première visite d'un nœud profond et sa revisite à l'itération
suivante est plus long que pour un nœud proche de la racine. Ça donne
au GPU une fenêtre de calcul alignée avec la probabilité de revisite :
les nœuds les plus utiles à précalculer sont justement ceux qui ont le
plus de temps avant d'être redemandés.

## Structure de cache GPU (séparée de la TT)

- Ne pas modifier `TTEntry` (16 octets, déjà bien packé, voir
  `src/engine/tt/transp_table.hpp`).
- Table indépendante, clé = zobrist key, valeur = score CNN +
  timestamp/génération.
- Éviction par âge simple (entrées plus vieilles que quelques
  itérations/secondes considérées obsolètes, la position racine ayant
  changé entre deux coups joués).
- Lecture non-bloquante uniquement (pas de wait, pas de future
  synchrone) au moment du probe TT principal.

## Sélection des positions à envoyer (filtre)

Ne pas envoyer systématiquement tous les nœuds visités à la
profondeur N — la queue GPU serait noyée de positions qui ne seront
jamais revisitées (branches coupées, coups dominés par le move
ordering à l'itération suivante). Filtre retenu :

1. Nœuds sur la PV courante (quasi certain d'être revisités).
2. Nœuds avec un score TT `EXACT` et une depth élevée (signe de
   stabilité, bon candidat de transposition future).
3. Exclusion des nœuds coupés par une simple fail-low/fail-high
   superficielle.

## Choix de la stack GPU : Metal/MPS vs Core ML/ANE

Retenu : **Metal / Metal Performance Shaders Graph**, direct.

Raisons :
- Contrôle fin de la queue et du callback asynchrone
  (`MTLSharedEvent` / completion handler sur command buffer), utile
  pour un flux de requêtes sporadiques plutôt qu'un gros batch
  continu régulier.
- Pas de copie CPU/GPU grâce à la mémoire unifiée
  (`MTLResourceStorageModeShared`).
- L'ANE (Apple Neural Engine, via Core ML) est plus efficace en
  énergie mais optimisé pour du débit soutenu avec batches réguliers,
  pas pour des requêtes isolées à latence de retour minimale — moins
  adapté à notre pattern d'usage. Voir aussi le point thermique dans
  [risques-et-estimations.md](risques-et-estimations.md).

## Points d'intégration à identifier dans le code

- Boucle d'iterative deepening (fichier orchestrant les appels
  successifs à `negamax` par profondeur croissante) : point où
  déclencher la sélection et l'envoi des positions à la queue GPU.
- `negamax.cpp` : point où remonter de la profondeur N vers N-1, pour
  intercepter les nœuds à envoyer avant de continuer l'exploration.
- `transp_table.hpp` : point de `probe()`/`store()` où consulter le
  cache GPU séparé en lecture seule.

(À préciser une fois l'implémentation commencée — pas encore localisé
précisément dans cette phase de design.)
