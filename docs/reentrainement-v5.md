# Re-entrainement (fine-tuning) du reseau v5

Question : continuer l'entrainement de v5 sur des donnees plus recentes
apporte-t-il de la force ?

## Protocole

- nnue-pytorch, commit 4289208, features `Full_Threats+HalfKAv2_hm^`.
- Depart : poids de v5. lambda = 0,6, lr 1e-4, gamma 0,75, ranger21.
- 10 epoques de 20 M positions (donnees test80-2024, janvier) ; validation
  sur fevrier.
- Export (`serialize.py --ft-compression none`, meme description que v5) des
  epoques 1, 4 et 9.
- Matchs contre v5, meme binaire v6.2, 10+0.1, Hash 16, 1 thread, livre UHO,
  400 parties chacun, concurrence 40, serveur 128 coeurs.

Entrainement : 15 min sur une RTX 5090.

## Resultats

Perte de validation : v5 0,00387 ; apres l'epoque 1 : 0,00358 ; puis plateau
a 0,00356.

| epoque | bench | Elo contre v5 | V / D / N |
|---|---|---|---|
| 1 | 16634 | +9,6 +- 20,4 | 106 / 95 / 199 |
| 4 | 15894 | -5,2 +- 21,8 | 104 / 110 / 186 |
| 9 | 16596 | -4,3 +- 21,2 | 104 / 109 / 187 |

**Neutre** : aucune epoque ne se distingue de v5. La baisse de la perte de
validation traduit une adaptation a la distribution des nouvelles donnees, pas
une meilleure evaluation en jeu. Non retenu.

## Incident

Premier lancement : les trois matchs en meme temps (3 x 40 parties, 120
moteurs chargeant chacun 111 Mo au demarrage) ; un moteur n'a pas repondu a
`uciok` a temps et fastchess a interrompu les tournois. Relance en serie, sans
probleme.

## Suite possible

Un gain de reseau demanderait plutot un entrainement long (plus de donnees,
ou architecture plus grande), de l'ordre de plusieurs heures de GPU.
