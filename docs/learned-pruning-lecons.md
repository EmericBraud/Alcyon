# Elagage appris : ce qu'on a appris

Toutes les lecons du projet, rangees par theme plutot que par date.
Chiffres et details : `docs/learned-pruning-bilan.md` (resume) et
`docs/learned-pruning.md` (journal). Branche `learned-pruning`, non fusionnee.

## Verdict

Trois formes testees, trois echecs en parties reelles :

| forme | meilleur resultat hors ligne | en parties |
|---|---|---|
| couper (rendre la borne sur prediction) | precision 97-99.9 % | SPRT 53 : **-63 Elo** |
| reduire de R plys, verifie ou gradue | 1 a 3 plies achetes a budget egal | chaque ply achete vaut 2.5-3x moins qu'un vrai |
| sonde a depth - 2 puis MoE "puis-je m'y fier ?" | 0.069 % d'erreurs a 30 % d'economie | **-27 Elo** (4000 parties) |

Un reseau predit tres bien l'issue d'un noeud. Mais les rares noeuds ou il
se trompe sont ceux qui decident du coup joue, et les elagages a la main
(RFP, NMP, LMR) ont deja pris les noeuds faciles.

## Sur l'elagage

1. **L'existant a deja consomme l'information "eval".** Pres des feuilles,
   84-90 % des noeuds sont coupes avant toute recherche (RFP surtout). Ceux
   qui restent sont ceux ou l'eval ne tranche pas : 11 a 91 % de fail-high,
   jamais mieux. Un modele ne peut gagner qu'avec de l'information hors eval
   (TT, accumulateur NNUE).
2. **La TT porte l'essentiel du signal.** Parmi les features scalaires,
   entree TT, borne, profondeur et score comptent le plus.
3. **La precision par noeud est la mauvaise mesure.** Un predicteur a
   99.5 % se trompe sur les 0.5 % qui surprennent, typiquement une
   tactique. Or la recherche existe justement pour trouver ces surprises.
4. **Couper sans verifier est fatal.** Couper 1 % des noeuds au hasard
   suffit deja a degrader. Toute reduction doit re-chercher quand le
   resultat surprend, comme la LMR.
5. **Le modele choisit mieux que le hasard (2 a 4x), mais pas assez.** A
   budget egal, le hasard achete des plies qui valent 0.4-1.0 point de
   perte chacun. Le modele fait 0.22-0.25, alors qu'un vrai ply "rapporte"
   environ 0.12. Il aurait fallu diviser la perte par ply par ~3.
6. **Un MLP sur l0 bat la regression hors ligne (-18 % de loss), pas en
   recherche.** A taux de declenchement egal, regression, MLP et hasard
   sont dans le bruit l'un de l'autre. 16 neurones suffisent ; 64
   surapprend.
7. **MoE par profondeur x cote x phase : c'est ce qui marche le mieux hors
   ligne.** Il fait 3.5x moins d'erreurs que la regle simple, et des seuils
   par expert (Lagrangien par bucket) gagnent encore un peu. Les experts
   convergent vite ; sur-echantillonner les faibles profondeurs aide.
8. **Une sonde coute plus que ses noeuds.** En partie, la TT est conservee
   et la table est petite. La sonde y coute environ 2x plus que sur le
   bench, et ce surcout se compose le long de l'arbre : +7 % de noeuds a
   d10, +18 % a d12. Le mecanisme se degrade donc a cadence plus longue.
9. **Les confiances du MoE regagnent les noeuds (-18 % a d12), mais leurs
   erreurs coutent ~40 Elo.** Quand on a deja une sonde, le probleme n'est
   plus le cout mais les erreurs.
10. **Les effets de bord de la sonde aident plutot.** L'isoler de
    l'historique et des killers coute 11 Elo. Lui interdire d'ecrire dans
    la TT ne change rien. L'emboitement (sonde dans sonde) n'explique
    qu'une petite part du cout.

## Sur la methode

1. **Evaluer a budget egal, symetrique.** Meme binaire et meme budget de
   noeuds par position pour la base et la config. Une base limitee en
   noeuds qui perd son iteration (98 % des noeuds) donnait +2 points
   factices a toutes les configs.
2. **Mesurer la qualite du coup, pas l'accord.** On evalue le coup joue
   par une recherche de la position suivante, avec E logistique. Un coup
   different mais aussi bon ne coute rien.
3. **Une reference ne doit pas prolonger la base.** ref14 repassait par
   les iterations 1..12 d'off12 : toute perturbation semblait perdre 10
   points. Stockfish comme reference mesure son style (60 % d'accord au
   plus) : rejete.
4. **Des controles a chaque mesure.** Config identique (doit donner 0
   exactement), Hash=4 (perturbation neutre), hasard au meme regime
   (plancher), base x2 / x4 (la mesure doit recompenser la profondeur).
5. **Comparer a taux de declenchement egal, pas a seuil egal.** A seuil
   egal, le MLP declenche 1.5 a 4x plus.
6. **Le bon critere : ecart de perte < 0 a budget egal.** Le critere
   "< 0.09 par ply" etait faux.
7. **3 000 positions ne suffisent pas.** Un gain de -0.09 sur 3 000
   positions est devenu +0.03 sur 12 000.
8. **Le bench ne voit pas le cout reel.** TT vide, fenetre complete a la
   racine, positions du binpack : c'est un autre contexte que la partie.
   En partie : TT depth >= depth 16.5 % contre 10 %, |eval - beta| median
   156 contre 107. Pour les couts, rejouer des parties avec la TT
   conservee (`ttd.py`).
9. **Il faut un match en parties avant de croire un gain de banc.** Le
   MoE etait >= 0 sur 12 000 positions et perd 27 Elo en parties.
10. **Un temoin inerte isole la cause.** Meme binaire, memes options,
    mecanisme sans effet (+1 +- 6) : la perte venait bien des sondes, pas
    de l'UCI ni de la gestion du temps. A faire des qu'une perte resiste
    aux hypotheses.
11. **Une experience ne vaut que si sa reponse peut changer la suite.**
    Avant un calcul long, se demander ce qu'on ferait de chaque issue.
12. **Verifier la parite C++/Python, et verifier la verification.** Une
    parite qui lit la meme entree des deux cotes ne voit pas un l0
    perime : `ALCYON_L0_CHECK` compare a un recalcul complet (0
    divergence). La parite du MoE se fait en `Threads=1`, car le tampon l0
    du dump est global et produit de faux ecarts a 10 threads.
13. **Le label compte.** P(fail-high) est un proxy. Pour une reduction
    verifiee, le bon label est "le reduit tombe du meme cote de beta que
    la recherche complete". Il a ameliore le modele, mais n'a pas suffi.
14. **Mesurer l'erreur la ou elle compte.** L'impact d'une erreur se mesure
    sur la decision finale, et doit tenir compte du budget libere pour les
    iterations suivantes.
15. **Un effet de bord non voulu change l'arbre.** Remplir le cache d'eval
    changeait `improving` partout (bug corrige, sans effet mesurable ici).
    Test simple : mecanisme sans action = bench identique a la base au
    noeud pres.

## L'erreur de methode, et le predicteur qui en sort

**Diagnostic** (`cost_errors.py`, 20 000 positions de parties, d12, TT
videe ; etalonnage : base d11 contre d12) :

| a d12 | coup change | pires | meilleurs | perte nette / position |
|---|---|---|---|---|
| base d11 (un ply de moins) | 11.4 % | 835 | 362 | +0.0024 |
| MoE actif | 22.6 % | 1 040 | 947 | +0.0016 (0.67 ply) |
| mode ombre | 19.3 % | 753 | 928 | -0.0004 (-0.17 ply) |

1. **Le gain avait ete mesure dans un contexte ou il est gonfle.** Offline,
   au bench et dans la qualite du coup, la TT etait vide au depart. La
   sonde y prouve a bas prix ce qu'en partie la TT sait deja. En partie, le
   gain en noeuds est environ moitie moindre, et le cout en qualite reste
   le meme.
2. **La decision a la racine est chaotique.** Le MoE change le coup deux
   fois plus souvent que retirer un ply, mais presque autant en mieux
   qu'en pire. La perte est un petit biais sous un grand bruit. Seuls 191
   des 630 cas pires ont un coupable unique, dont 61 % de vraies erreurs au
   sens du label (1.3 % dans le temoin). Ces coupables ont un z plus pres
   du seuil, une eval plus proche de beta et plus souvent une entree TT :
   une tendance, pas une signature. Aucune mesure noeud par noeud ne peut
   voir ce biais.
3. **Le predicteur en plies :** solde = economie en partie (`ttd.py`,
   noeuds a d12, EBF de partie 1.54) - cout en qualite a profondeur egale
   (`cost_errors.py`, en plies grace a l'etalonnage d11/d12). Environ 5
   minutes par variante.

| config | qualite (plies) | economie (plies) | solde predit | Elo mesure |
|---|---|---|---|---|
| mode ombre | +0.17 | -0.38 | -0.2 | -23 |
| MoE actif | -0.67 | +0.47 | -0.2 | -27 |

Les deux points concordent par des chemins opposes (+-0.13 ply par terme),
ce qui donne environ 100 Elo par ply a 5+0.05. Toute variante se juge par
ce solde avant un match.

## Plafond au niveau du coup

Oracle des coups calmes tardifs (`ALCYON_ORACLE_QUIET`, etape QUIETS du
MovePicker : apres coup TT, bonnes prises, killers, contre-coup). Il
supprime, dans l'arbre tel quel, chaque coup qui ne monte pas alpha. Mesure
sur 345 parties rejouees, TT conservee :

| profondeur | retirable, tous noeuds | retirable, non-PV seuls | coups calmes tardifs inutiles |
|---|---|---|---|
| d10 | 92.8 % | 64.7 % | 96.16 % |
| d12 | 94.7 % | 73.2 % | 96.16 % |

C'est l'ecart a l'arbre minimal, pas un gain atteignable : l'oracle sait
d'avance. La difficulte, ce sont les 3.8 % de coups utiles, que la LMR et
la LMP traitent deja par rang. Mais le levier est la : les sous-arbres de
coups calmes tardifs non-PV font environ 73 % de l'arbre, et 10 % de cette
part valent environ 0.2 ply. Au niveau du noeud, l'existant avait deja tout
pris.

## Au niveau du coup : resultats

Dump des coups calmes tardifs (`ALCYON_QUIET_DUMP`, profondeurs equilibrees
par `ALCYON_QUIET_DUMP_P`, l0 de la position apres le coup), MLP l0 a 12
experts (`quiet_mlp.py`), table profondeur x rang, export et parite
(`quiet_export.py`), decision = +1 ply de LMR sous un seuil de P(utile) /
cout attendu (`learned_quiet_mode`).

Hors ligne, l0 aide aux faibles profondeurs (d1-3 : 24 % du cout retirable
a 1 % de perte, contre 19 % en scalaire et 8.5 % pour le rang). Mais dans un
classement global, la table profondeur x rang fait presque tout (45 %),
parce que l'essentiel du retirable est aux grandes profondeurs.

| variante | solde predit | Elo (4000 parties) |
|---|---|---|
| table, 0.1 % | +0.30 ply | -1.9 +- 6.8 |
| table, 1 % | +0.22 ply | -5.3 +- 6.8 |
| MLP l0 d<=6, 0.1 % / 1 % | +0.18 / -0.12 ply | (pas de match) |
| hybride, 0.1 % / 1 % | +0.16 / -0.04 ply | (pas de match) |

1. **Le predicteur en plies a un biais optimiste d'environ 0.3 ply.**
   Quatre points : -0.2 predit -> ~-25 Elo, +0.2 a +0.3 -> ~0. Il sert a
   classer et ecarter des variantes, pas a promettre un gain : viser au
   moins +0.5 predit avant un match.
2. **La table dit "plus de LMR aux grandes profondeurs"** (presque tous les
   coups calmes tardifs a d >= 7-10), et c'est neutre en parties : la LMR
   reglee par SPSA est deja pres de l'optimum.
3. **Dans la recherche, le MLP l0 ne fait mieux ni que la table ni que la
   base.** Aux faibles profondeurs, une reduction se paie en re-recherches
   pour peu d'economie (+3.9 % de noeuds).
4. **Pieges du dump au niveau du coup :** lire l'historique APRES la
   recherche du coup fuit le label (sa coupure l'a deja augmente) ; ecrire
   la sortie du moteur au debut du coup et l'enregistrement a la fin les
   desaligne (coups imbriques) ; a d12 fixe, profondeur restante et ply sont
   confondus (d10 = ply 2), donc la table ne dit pas lequel des deux porte
   l'effet ; elle est vide a d13+.

## Sur le moteur (bugs trouves en chemin)

Tous trouves en chassant des anomalies de mesure :

| bug | symptome | etat |
|---|---|---|
| pas de mate distance pruning | 33M noeuds a d11 sur une finale, 9 787 apres | dans `main`, SPRT +2.7 |
| extensions d'echec sans fin | finales de dames : seldepth ~2x, 100M+ noeuds | extension retiree, SPRT +3.3 |
| boucle infinie de l'aspiration | score "presque mat" venu de la TT : `go depth` ne rendait jamais la main | dans `main` |
| course sur le plateau UCI (ponder, `position`) | coups rejetes, `free(): invalid size`, 192 blocages sur 345 parties, probablement les 112 coups illegaux en tournoi | dans `main`, 345/345 parties rejouees OK |
| bruit d'ordonnancement identique entre threads Lazy SMP | `(thread_id << 32) & 0x7FF` vaut toujours 0 | non corrige, a tester en SPRT multi-thread |

Les correctifs aspiration et UCI sont fusionnes sans SPRT (bench
identique). Un SPRT de non-regression reste a faire.

## Sur l'infrastructure

1. **Enregistrer la sortie dans un fichier et la relire**, plutot que
   relancer une commande longue.
2. **`pkill -f` peut tuer sa propre session ssh** si le motif apparait dans
   la ligne de commande. Utiliser `pkill -x`, ou un motif du type `[x]yz`
   absent de la commande.
3. **Ne pas modifier un script shell pendant qu'il tourne** : `sh` le lit
   au fil de l'eau.
4. **Ne pas recompiler le binaire d'un match en cours** : utiliser un
   build separe.
5. **zsh ne decoupe pas `$var`** (`set -- $cfg`) : passer par `bash -c`.
   Un glob sans correspondance y interrompt une chaine `&&`.
6. **Des drivers Python orphelins a 100 % CPU** ont produit 198 faux
   timeouts. Detecter la fin de flux (readline vide) et tuer les restes
   avant chaque serie.
7. **Occuper les 192 coeurs** avec des files dynamiques, et comparer les
   dumps avec un seul worker (sinon le bruit d'echantillonnage domine).
8. **Surveiller le disque** : dumps l0 de 1 Ko par noeud, donc des dizaines
   de Go.

## Ce qui reste

- **Au niveau du coup, sur les coups calmes tardifs** (plafond ci-dessus).
  Dump des coups avec leurs features (rang, historiques, SEE, echec, eval
  - alpha) et le label "utile". Puis part des noeuds retirables a 99 % de
  rappel des coups utiles, et integration en modulation de la LMR (+-1
  ply), jugee par le predicteur en plies.
- **Moduler l'existant au lieu d'ajouter.** Utiliser la sortie du reseau,
  sans recherche, pour decaler de +-1 ply la LMR, ou la marge du RFP ou du
  NMP. Une erreur n'y coute qu'un ply de reduction, et ces mecanismes sont
  deja regles et verifies. C'est la seule piste non essayee.
- Correction history, et LMR apprise au niveau du coup.
- SPRT de non-regression des correctifs de robustesse dans `main`.
- Correctif du bruit Lazy SMP.
- Avant toute fusion de la branche : `learned_prune_enabled` a 0 par
  defaut.
