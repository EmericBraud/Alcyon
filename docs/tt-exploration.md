
## Eval statique memorisee en TT (branche `tt-eval`, `tt_eval_cache`)

L'eval NNUE brute (avant correction history) du camp au trait est stockee
dans les 16 bits hauts du champ cle ; la cle est verifiee sur 48 bits (l'index
du paquet vient des bits bas), donc l'entree garde 16 octets. Negamax et
quiescence relisent l'eval en TT avant d'appeler le reseau.

- Bench : 16763 noeuds avec et sans cache. L'eval relue est exactement celle
  qu'on aurait recalculee, la recherche n'est pas modifiee.
- Vitesse (Mac, 8 positions UHO, profondeur 16, deux passes) : ~494 k nps sans
  cache, ~535 k nps avec, soit **+8 % de noeuds par seconde**.

A tester en parties (gain attendu : quelques Elo, a toute cadence).
