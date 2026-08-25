# Eval GPU asynchrone (CNN) en complément du NNUE

Branche : `gpu-eval-async`

## Idée

Faire cohabiter le NNUE actuel (CPU, hot path de la recherche) avec un
réseau CNN plus gros et plus précis exécuté sur le GPU (Apple Silicon,
mémoire unifiée), **sans jamais bloquer le thread de recherche CPU**.

Le CNN ne remplace pas l'éval NNUE. Il fonctionne en tâche de fond et
vient corriger/affiner l'éval de positions déjà vues, à la prochaine
fois qu'elles sont atteintes (transposition ou itération suivante de
l'iterative deepening).

## Documents

- [architecture.md](architecture.md) — pipeline détaillé, structure de
  cache, point d'intégration, choix Metal vs Core ML/ANE.
- [risques-et-estimations.md](risques-et-estimations.md) — pourquoi
  l'intégration directe dans l'arbre alpha-beta est dangereuse, et
  estimation du gain Elo attendu.

## Décision de design retenue

1. Jamais d'attente bloquante du résultat GPU dans la recherche.
2. Résultat GPU stocké dans une structure séparée de la TT principale
   (`zobrist_key -> {cnn_score, timestamp}`), pas dans `TTEntry`.
3. Envoi des positions vers la queue GPU en remontant l'arbre à la fin
   de l'exploration d'une profondeur N (pas en fin d'itération ID) —
   ça maximise le temps disponible avant la revisite probable à N+1.
4. Sélection des positions envoyées : PV actuelle + nœuds
   EXACT/depth élevé. Pas d'envoi systématique de tous les nœuds.
5. Implémentation GPU via Metal/MPS direct (pas Core ML/ANE) — mieux
   adapté à des requêtes sporadiques à faible latence de retour,
   contrôle plus fin de la queue et du callback asynchrone.

## Statut

Discussion de design uniquement à ce stade. Pas de code écrit.
