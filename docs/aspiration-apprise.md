# Fenetre d'aspiration apprise

Branche `asp-mlp`. Remplace la largeur fixe de la fenetre d'aspiration
(+-16 / 41 / 20 selon la profondeur) par une largeur predite a chaque
iteration : `delta = c x exp(prediction)`, ou la prediction est
log(1 + |score(d+1) - score(d)|).

## Pourquoi c'est un bon candidat

Comme la gestion du temps : un appel par iteration (cout nul), une cible
propre (la variation du score a l'iteration suivante, observee directement).
Une fenetre trop etroite coute une re-recherche ; trop large, elle elague
moins.

## Constat hors-ligne (1,15 M iterations, donnees de la gestion du temps)

- la fenetre actuelle est ratee 22 % du temps ;
- quantiles de |variation| : mediane 0, q75 20, q90 62, q95 99, q99 204 ;
- un predicteur (MLP 32x16, 9 signaux de recherche) explique ~la moitie de
  la variance de log(1 + |variation|) ; a largeur moyenne egale (~25), la
  fenetre apprise est ratee 17 % du temps au lieu de 22 % ; a couverture
  egale (85 %), elle est 25 % plus etroite.

## Accumulateur de la racine

Le l0 de la racine (1024 octets) est projete en 16 valeurs **une fois par
coup** (l'accumulateur est deja tenu a jour de facon incrementale ; seule la
projection est calculee), puis concatene aux 9 signaux a chaque iteration
(16 + 9 -> 32 -> 1). Donnees : 908 k iterations avec leur l0 (cadences
melangees 5+0.05 a 60+0.6).

| Modele | MSE (log) | couverture / largeur moyenne (c = 3) |
|---|---|---|
| variance de la cible | 3.126 | |
| sans l0 (meme architecture, l0 a zero) | 1.740 | 81.3 % / 24.5 |
| **avec l0** | **1.708** | 81.8 % / 25.6 |

Gain reel mais modeste : la "texture" de la position apporte ~2 % d'erreur
en moins par rapport aux seuls signaux de recherche.

## Parties (auto-jeu contre v6.1, 10+0.1, 500 parties)

| Reglage | Elo |
|---|---|
| c = 3 | +12.5 +- 18.1 |
| c = 2 | +16.0 +- 18.7 |
| c = 2 (2e serie) | (en cours) |
| c = 2, 40+0.4 | (en cours) |
