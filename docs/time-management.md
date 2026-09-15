# Gestion du temps : mesures

## Pourquoi cette page

Le canal GPU (valeur, puis ordonnancement) a ete ferme par la mesure a
+/- 2 Elo, cf. `docs/gpu-async-eval/consultative-eval-measurements.md`.
La gestion du temps a ete retenue comme piste suivante parce que l'effet
attendu depasse le bruit d'un match de quelques centaines de parties. Cette
page garde les chiffres, pour ne pas re-deviner plus tard ce qui a ete
mesure.

## Stabilite du coup racine (300 positions, profondeur 12)

300 ouvertures a 8 plis filtrees par eval, une recherche mono-thread par
position, 3598 lignes `info` collectees.

Taux de changement du meilleur coup d'une iteration a la suivante :

| transition | taux |
|---|---|
| 8 -> 9 | 16,0 % |
| 9 -> 10 | 14,7 % |
| 10 -> 11 | 11,3 % |
| 11 -> 12 | 10,1 % |

Pouvoir predictif de la duree de stabilite (2398 echantillons, taux de base
18,1 %) :

| iterations stables | P(changement) | n |
|---|---|---|
| 0 | 31,1 % | 469 |
| 1 | 28,4 % | 394 |
| 2 | 19,5 % | 302 |
| 3 | 16,3 % | 313 |
| 4 | 9,8 % | 245 |
| 5 | 10,5 % | 200 |

Monotone, 3x d'ecart entre les extremes. C'est de la que ces cinq facteurs
`StabilityFactor0..4` de `config.hpp` sortent.

### Le piege : 3x d'ecart n'est pas 3x d'information

Sur les 1923 echantillons des six classes, la feature n'explique que ~4 %
de la variance -- trois metriques independantes concordent :

| mesure | sans | avec | explique |
|---|---|---|---|
| Brier | 0,16855 | 0,16182 | 3,99 % |
| entropie | 0,7502 bits | 0,7200 bits | 4,02 % |
| r^2 (corr. -0,195) | | | 3,81 % |

Les deux chiffres sont vrais ensemble : meme dans la pire classe on se
trompe 31 % du temps, dans la meilleure 10 %. Il n'y a jamais de certitude,
seulement un deplacement de moyenne.

Et c'est pourquoi 4 % ont suffi : on ne parie pas sur une decision isolee ou
l'erreur coute la partie, on regle un budget des milliers de fois et seule
l'esperance compte. C'est l'inverse exact du canal elagage, ecarte parce
qu'une erreur y corrompt le resultat.

Consequence pour un eventuel reseau : la barre a battre est une feature
gratuite qui separe deja 31 % de 10 % de facon monotone. Les 96 % de
variance restants sont en grande partie irreductibles -- savoir si le coup
changera a la profondeur 13 demande d'avoir fait la recherche de profondeur
13.

## Validation Elo

`f91761b` (soft/hard limits + `Move Overhead` lu + `movestogo` parse +
table de stabilite) contre `035d647` (juste avant, limite unique).

Dispositif : 10+0.1, 1 thread, Hash 128, `OwnBook=false` (sinon le livre
court-circuite la recherche), GPU eval off des deux cotes, 400 ouvertures
8 plis filtrees par eval, couleurs inversees, concurrency 3 sur 4 P-cores,
secteur.

```
Games: 216, Wins: 81, Losses: 24, Draws: 111, Points: 136.5 (63.19 %)
Elo: +93.91 +/- 29.02, nElo: 157.80 +/- 46.33, LOS: 100.00 %
Ptnml(0-2): [0, 12, 38, 47, 11], PairsRatio: 4.83
```

Trajectoire stable, sans la regression qu'on pouvait craindre d'un debut de
match : +84,9 (96 p.) -> +91,6 (128 p.) -> +96,8 (184 p.) -> +93,9 (216 p.).

`Ptnml[0] = 0` : sur 108 paires, jamais deux defaites dans la meme paire.

### Ce que ce chiffre ne dit pas

Le +94 est **non decompose**. Trois mecanismes y sont melanges, et deux
d'entre eux sont inactifs ou negligeables a cette cadence :

- `movestogo` : 10+0.1 est une mort subite, fastchess n'envoie jamais ce
  champ. Contribution nulle ici. Le correctif ne se verifie qu'en cadence
  par periodes (`tc=40/60`).
- `Move Overhead` : retire 100 ms sur 10 000, soit 1 % du budget, et zero
  perte au temps observee dans les deux camps. Contribution negligeable.
- Donc le +94 vient quasi entierement du **soft/hard limit** et de la
  **table de stabilite**, sans qu'on sache le partage.

Hypothese a tester (ablation `StabilityFactor0..4` tous a 1,0 contre
`f91761b`) : l'essentiel vient du soft/hard limit, la table valant de
l'ordre de 10 Elo. Tant que ce n'est pas mesure, les cinq facteurs ne
meritent pas de SPSA.
