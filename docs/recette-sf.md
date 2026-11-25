# Recette d'entraînement des réseaux Stockfish (relevée le 25/11/2026)

Sources : wiki de nnue-pytorch (`official-stockfish/nnue-pytorch.wiki`, pages « Training
datasets » et « Training workflows ») et la recette de l'architecture à menaces,
`vondele/nettest/threats.yaml` (branche master), vers laquelle le wiki renvoie.

## Données

- Environ 60 binpacks, **tous réévalués par Leela BT4** (`relabel-BT4-tf13tune`), le réseau
  transformeur de Leela, bien plus fort que les réseaux qui jouaient les parties test80 :
  - parties Leela T60 à T80 (2021 à février 2024, filtrées par linrock), les mêmes mois que
    les nôtres mais avec des scores BT4 (`vondele/linrock_relabel_1` et `_2`,
    `xushawn/test80-bt4-relabel`, environ 540 Go) ;
  - `leela96`, `T60T70wIsRightFarseer...`, `dfrc_n5000` (Fischer aléatoire),
    `nodes5000pv2_UHO`, `wrongIsRight`, `multinet` ;
  - `sf-searchtree-n200k` : **positions tirées des arbres de recherche de Stockfish**, réévaluées
    par BT4 (`xushawn/sf-searchtree-bt4`, 1,1 To) ;
  - `tb5dtm` : finales de tables de 5 pièces.
- Le wiki note qu'entraîner uniquement sur des données Leela depuis zéro donne de moins bons
  réseaux : couverture des positions, ou concepts trop durs à apprendre sans base.

## Entraînement

- Trainer : fork `xu-shawn` de nnue-pytorch (sha 289991e), pas la version officielle.
- Features : `Full_Threats+PP_3Wide+K32H3^` (plus récentes que notre `HalfKAv2_hm`),
  L1 1024, L2 32, piles de couches choisies par nombre de pièces, dames, pions.
- AdamW, lr 4e-4 en « one cycle » (5 % de montée, division finale par 1000), lots de 131 072
  positions, époques de 2^30 positions (1,07 milliard).
- **Pré-entraînement depuis zéro : 560 époques, répété 3 fois** (environ 600 milliards de
  positions par répétition), puis **affinage de 600 époques** (lots de 65 536). Au total
  de l'ordre de 1 200 milliards de positions. À titre de comparaison, notre run long en
  a vu 16 milliards.
- Lambda à 1,0 (évaluation pure, avec un léger bruit aléatoire), décroissance des poids,
  taux d'apprentissage réduits pour les features de menaces (× 0,5) et les buckets du
  feature transformer (× 0,25), pondération de la perte selon le ply et le nombre de pièces,
  saut des 18 premiers plies.

## Ce qu'on en retient pour Alcyon

1. **Qualité des scores** : les mêmes parties test80 que les nôtres existent réévaluées par
   BT4. C'est toujours Leela, mais un Leela beaucoup plus fort. Test direct et peu coûteux :
   fine-tuning sur ces versions réévaluées.
2. **Positions de recherche** : SF entraîne sur des positions de ses propres arbres de
   recherche, celles où l'éval sert réellement en partie. C'est la réponse à l'écart mesuré sur
   les positions déséquilibrées.
3. **Échelle** : environ 75 fois plus de positions que notre run long. Hors de portée de
   notre budget en l'état (des semaines de 5090).
