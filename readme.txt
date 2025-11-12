# ♟️ Chess 26
Moteur d'Échecs Haute Performance en C++

> Moteur en cours de développement !!!

## 💡 Présentation du Projet
Chess 26 est un projet ambitieux visant à implémenter un moteur de jeu d'échecs (intelligence artificielle) complet et performant. Le but principal de ce projet est de fournir un programme capable de jouer aux échecs contre un utilisateur avec une efficacité et une vitesse d'exécution maximales.

Ce projet sert également de vitrine concrète pour démontrer ma maîtrise du langage C++ moderne et des techniques d'optimisation (algorithmique et bas niveau) nécessaires pour répondre aux exigences de performance très élevées de ce domaine.

## ⚙️ Technologies et Compétences Mises en Avant
Ce moteur est entièrement développé en C++ et utilise des techniques avancées pour garantir une rapidité d'exécution optimale.

Langage : C++ (20)

Performance : Utilisation intensive des fonctionnalités C++ de bas niveau pour l'efficacité mémoire et l'optimisation CPU.

Représentation du Plateau : Implémentation de la structure Bitboards pour une manipulation rapide et efficace de l'état du jeu et du calcul des mouvements.

Algorithme de Recherche : Implémentation de l'algorithme Minimax avec élagage Alpha-Beta (Alpha-Beta Pruning).

Optimisations :

Transposition Table pour éviter la répétition des calculs.

Heuristiques avancées (Killer Moves, History Heuristic).

Gestion des mouvements légaux et des règles spéciales (roque, promotion, prise en passant).

Tests : Mise en place de tests unitaires pour valider l'exactitude des mouvements et de l'algorithme.

## 🚀 Fonctionnalités Désirées
Jeu Contre l'IA : Possibilité de jouer une partie d'échecs complète contre l'ordinateur.

Notation Standard : Support de la notation FEN (Forsyth-Edwards Notation) pour charger/sauvegarder des positions.

Interface Console : Interface utilisateur en ligne de commande pour la phase initiale de développement et de test.

Évaluation de Position : Fonction d'évaluation statique prenant en compte le matériel, la position et la structure des pions.

## 🛠️ Compilation et Exécution
Prérequis
Un compilateur C++ compatible C++20 ou supérieur (ici g++).


## Instructions
Cloner le dépôt :

```Bash
git clone https://github.com/EmericBraud/chess26.git
cd chess-26
```

Créer le répertoire de build et compiler avec le Makefile :

```Bash
make all
```
Lancer le programme : L'exécutable généré se trouvera dans le répertoire principal.

```Bash
./chess_26
```

# 📝 Licence
Ce projet est sous licence MIT.

# 👨‍💻 Auteur
Emeric Braud - https://www.linkedin.com/in/emeric-braud-101239151/