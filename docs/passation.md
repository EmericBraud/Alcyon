# Passation : etat du moteur, infrastructure, methode, suite

Document de reprise pour un autre agent (ou un autre PC). Ecrit le
2026-09-30. Detail des experiences d'elagage appris :
`docs/learned-pruning-lecons.md` et `docs/learned-pruning-bilan.md` sur la
branche `learned-pruning`.

## 1. Etat de `main` (b0890cc)

Trois gains valides le 2026-09-29, environ +30 Elo en 8+0.08 (addition des
mesures, le cumul n'a pas ete mesure directement) :

| commit | changement | validation |
|---|---|---|
| 8e15e5c | **historiques conserves d'un coup a l'autre** (`persist_history`, vieillissement /2) | SPRT [0,3] accepte : +17.5 +- 6.2, 4 316 parties, 8+0.08 |
| 7bb6245 | **correction history** par structure de pions (eval statique + stand pat), conservee avec les historiques | valide sur decision de l'utilisateur : +3.4 +- 3.1, 17 343 parties, LLR 1.79 / 2.94 |
| b0890cc | **contre-coup retabli** (mise a jour perdue dans 60eb372, V 4.0.1) | SPRT [0,3] accepte : +9.1 +- 4.4, 8 806 parties |

Plus tot (deja dans `main`) : mate distance pruning (51e170f), pas
d'extension d'echec (12b8030), boucle d'aspiration (3d4ec32), course sur le
plateau UCI (37d4d60). SPRT de non-regression [-3,1] de ces deux derniers
accepte : +1.3 +- 2.3, 27 414 parties, 0 incident cote corrige (l'ancienne
version : 4 deconnexions, 1 coup illegal, 454 avertissements PV).

Bench de `main` : 16377.

**La cause du plus gros gain** : `EngineManager::start_workers` recree les
`SearchWorker` a chaque "go", et leur constructeur appelle
`clear_heuristics()`. Chaque coup repartait donc d'historiques vides. La
copie `SavedHeuristics` (allouee une fois dans `EngineManager`) est
rechargee au depart et sauvee depuis le thread 0 a la fin ; `ucinewgame` l'efface.

## 2. Branches

Toutes poussees sur `origin` (github.com:EmericBraud/alcyon). Worktrees
locaux dans `~/Desktop/code/chess26-*` (a recreer sur un nouveau PC).

| branche | contenu | etat |
|---|---|---|
| `main` | voir section 1 | reference |
| `persist-history`, `corr-hist`, `counter-move` | les trois gains | fusionnees dans `main` |
| `cont-malus` | malus sur les historiques de continuation | SPRT arrete : -2.0 +- 4.4 (9 145 parties), rejete |
| `lmr-history` | LMR : r -= clamp(historique / 1024, -2, 2) | SPRT arrete : -3.2 +- 4.4 (7 981 parties), rejete ; diviseur non regle (candidat SPSA) |
| `corr-np` | correction history hors pions (2 tables de plus) | SPRT arrete a 1 775 parties (-2.3 +- 10.3), **non conclu** |
| `hist-prune` | elagage des coups calmes a tres mauvais historique (d <= 3) | **jamais teste** |
| `learned-pruning` | toute l'exploration (elagage appris, MoE, oracles, dumps, outils) | branche d'etude, **ne pas fusionner** (defaut `learned_prune_enabled=1`) |
| `robustness-fixes`, `mate-distance-pruning`, `aspiration-mate-loop`, `uci-board-race` | anciennes branches deja fusionnees | historique |

Les branches `cont-malus`, `lmr-history`, `corr-np`, `hist-prune` sont
basees sur un `main` anterieur a b0890cc pour certaines : rebaser sur
`origin/main` avant tout test.

## 3. Infrastructure

### Serveur de calcul (AWS)

- Instance `i-0374658a6c5fe48a3`, 192 vCPU (Graviton, arm64), IP
  `16.16.201.208` (change a chaque redemarrage), utilisateur `ubuntu`.
- Cle SSH : `~/Downloads/chess26_tuning.pem` sur l'ancien PC. **A copier sur
  le nouveau PC** (hors depot).
- Volume `/data` (100 Go, persistant, monte via `/etc/fstab`, `nofail`).
  **Arreter l'instance quand elle ne sert pas** : le volume reste.
- Contenu utile de `/data` :
  - `/data/alcyon` : clone du depot (branche `learned-pruning` en general) ;
    builds `build-nott` (NNUE + SPSA, sert aux matchs d'options) et
    `build-exps` (NNUE + `ENABLE_SEARCH_EXPERIMENTS` : dumps, oracles).
  - `/data/alcyon/data` : fichiers du reseau NNUE (`ALCYON_DATA_DIR`).
  - `/data/Client/fastchess-ob` : fastchess ; livre `/data/Client/Books/UHO_4060_v2.epd`.
  - `/data/venv` : Python (torch, sklearn, python-chess).
  - `/data/ctx` : 345 parties du moteur (`game_NNN.txt` : un coup UCI par
    ligne) et `positions.txt` (48 440 positions) ; base de tous les rejeux.
  - `/data/sprt_<nom>` : worktrees + builds des SPRT (`sprt_cm` = `main` b0890cc).
  - Gros dumps (supprimables) : `dump_v3` (30 Go), `ctx` (16 Go, garder
    les `game_*.txt` et `positions.txt`), `qdump_bal2` (12 Go), `dump_v3s`,
    `odump2`.

### Tests

- **fastchess direct sur le serveur**, pas OpenBench (le tunnel ngrok vers
  l'OpenBench local du Mac, `127.0.0.1:8000`, utilisateur "local", n'etait
  pas actif ; l'utilisateur a choisi de rester sur fastchess).
- `tools/testing/sprt_gen.sh <branche> <binaire de reference> <nom>` : SPRT
  [0,3], 8+0.08, Hash 16, 180 parties en parallele, `-recover`. Construit la
  branche dans `/data/sprt_<nom>`, affiche le bench des deux binaires (a
  verifier !), journal `/data/sprt_<nom>.log`.
- `tools/testing/match.sh <binaire> <nom> [option=valeur ...]` : 4000 parties
  en 5+0.05, meme binaire, variante contre base ; `BASE_OPTS` pour les
  options communes. Environ 6 minutes sur 180 coeurs. Sert a trier avant un SPRT.
- Non-regression : `-sprt elo0=-3 elo1=1`.
- **Toujours `-recover`** : sans lui, fastchess arrete tout au premier moteur
  bloque.
- Commits : message en francais, ligne `Bench: N` pour les changements
  fonctionnels ; **pas de trailer Co-Authored-By**.

### Builds

    cmake -S . -B build-rel -DCMAKE_BUILD_TYPE=Release -DENABLE_NNUE_EVAL=ON
    cmake --build build-rel -j
    printf 'bench\nquit\n' | ./build-rel/alcyon     # bench

Options : `-DENABLE_SPSA_TUNING=ON` (parametres modifiables par UCI, sinon
`constexpr`), `-DENABLE_SEARCH_EXPERIMENTS=ON` (dumps et oracles par variables
d'environnement `ALCYON_*`). Le moteur lit le reseau dans `ALCYON_DATA_DIR`.

## 4. Ce qui a ete essaye (et ne marche pas)

Details et chiffres : `docs/learned-pruning-lecons.md` (branche `learned-pruning`).

| piste | resultat en parties |
|---|---|
| coupe apprise (regression) | SPRT -63 |
| reduction apprise, verifiee ou graduee | chaque ply achete vaut 2.5-3x moins qu'un vrai |
| sonde a depth-2 + MoE "puis-je m'y fier" (20 experts) | -27 ; mode ombre -23 ; reentraine sur parties -31 |
| MoE limite a une fenetre de profondeur | d3-4 : -23 ; d5-6 : -18 |
| +1 ply de LMR sur les coups calmes tardifs (table profondeur x rang) | -1.9 / -5.3 |
| idem avec MLP l0 (12 experts) | moins bon que la table au predicteur |
| ordonnancement par l'eval statique de l'enfant | -16 (l'eval ne voit pas la piece en prise ; la LMR en patit) |
| reseau d'ordonnancement sur l0 du noeud | reproduit l'ordre actuel |
| ProbCut | -4.7 +- 6.8 |
| modulation RFP / NMP par le modele de noeud (regression) | entre -3.3 et +3.5 |
| idem MLP, calcule a chaque noeud | -67 (26 % de nps) |
| idem MLP, calcule seulement ou utile | RFP etendu a d9 : **+3.4 +- 6.7** ; NMP R+1 : -19.4 |

**Conclusion** : les reseaux entraines hors ligne ne battent pas les
heuristiques reglees dans l'elagage. Les gains viennent de tables apprises
**en ligne** pendant la partie (historiques, contre-coup, correction).

## 5. Lecons de methode (a lire avant toute experience)

1. **Juger en parties.** Hors ligne, au bench ou nœud par nœud, les mesures
   ont regulierement menti. Le bench part d'une TT vide et d'historiques
   vides : il ne voit ni le cout reel d'une sonde (2x plus cher en partie)
   ni l'effet d'un changement d'ordonnancement.
2. **Rejouer des parties avec la TT conservee** (`tools/learned_pruning/ttd.py`,
   branche `learned-pruning`) pour mesurer noeuds a profondeur egale en
   conditions de partie. `cost_errors.py` mesure la qualite du coup a
   profondeur egale ; etalonnage d11/d12 = 0.0024 E par ply.
3. **Predicteur en plies** = economie (ttd) - cout en qualite. Il classe bien
   les variantes mais a un **biais optimiste d'environ +0.3 ply** : ne
   promettre un gain qu'au-dela de +0.5 predit.
4. **Temoin inerte** (memes options, mecanisme sans effet) des qu'une perte
   resiste aux hypotheses : il separe le cout du mecanisme de celui du
   code autour.
5. **Parite C++ / Python** pour tout modele exporte, dump fait en
   `Threads=1` (tampons globaux).
6. **Pieges des dumps** : lire les features AVANT la recherche du coup
   (l'historique lu apres contenait deja la coupure : fuite du label) ;
   ecrire la sortie du moteur avec l'enregistrement (les ecritures
   imbriquees se desalignent) ; ponderer par 1/p du tirage ; equilibrer les
   profondeurs (`ALCYON_QUIET_DUMP_P`) ; a `go depth 12` fixe, profondeur
   restante et ply sont confondus.
7. **Plafonds par oracle** avant de construire : ordonnancement parfait =
   -40.6 % de noeuds (~1.2 ply) ; coups calmes tardifs inutiles = 73 % de
   l'arbre (hors PV). Ce sont des ecarts a l'arbre minimal, pas des gains
   atteignables.
8. **Un SPRT peut s'arreter tot** sur decision de l'utilisateur (il l'a fait
   pour la correction history). Sinon, laisser conclure : regarder
   l'intervalle en cours de route gonfle les faux positifs.

## 6. Pieges d'exploitation

- `pkill -f <motif>` tue sa propre session ssh si le motif figure ailleurs
  dans la meme commande : ecrire `[x]` dans le motif et ne pas repeter le
  nom du script dans la commande.
- zsh ne decoupe pas `$var` (`for o in $opts`) : utiliser `${=var}` ou `sh`.
- Ne pas modifier un script shell en cours d'execution (lu au fil de
  l'eau) ; ne pas recompiler le binaire d'un match en cours (build separe).
- Enregistrer les sorties longues dans un fichier et les relire, lancer les
  attentes en arriere-plan.
- `caffeinate -dims` sur le Mac pour qu'il ne s'endorme pas pendant les
  longues series (ne vaut pas capot ferme sans alimentation).
- Surveiller le disque `/data` (85 % au 2026-09-30).
- Pas d'allocation en chemin critique (`std::vector` interdit hors
  allocations uniques au demarrage, comme la TT).

## 7. La suite, par ordre de priorite

1. **SPSA** sur LMR (table, CutNodeBonus, MinMovesSearched...), NMP (RConst,
   RDiv, MinDepth), RFP (marges, MaxDepth), futility, IIR. Ces parametres ont
   ete regles quand les historiques repartaient de zero a chaque coup ;
   l'ordonnancement a gagne ~30 Elo depuis. fastchess n'a pas de SPSA :
   ecrire un petit pilote (perturbation +-c, deux moteurs, quelques
   centaines de parties par iteration) ou passer par OpenBench (tunnel ngrok
   + worker sur le serveur). Gain le plus probable : +5 a +15.
2. **SPRT du RFP etendu a d9 module par le MLP** (branche `learned-pruning`,
   options `mod_on=1 mod_model=1 mod_rfp_depth=9`, gains du jour actives des
   deux cotes : `persist_history=1 corr_hist=1 counter_move_update=1`). Seule
   variante MLP legerement positive (+3.4 +- 6.7). S'il passe, porter le
   modele de noeud dans `main` (features, poids MLP, `fill_prune_record`).
3. `corr-np` et `hist-prune` : petits gains attendus (+1 a +5), marges a
   regler par SPSA plutot qu'au SPRT.
4. Ordonnancement par eval de l'enfant **avec le SEE de la case d'arrivee**
   (le defaut probable de la version a -16) : 20-30 % de chances.
5. Bruit Lazy SMP identique entre threads (`move_picker.hpp`,
   `(thread_id << 32) & 0x7FF` vaut toujours 0) : a corriger et tester en
   SPRT multi-thread.
6. Separer profondeur restante et ply : refaire le dump des coups calmes
   avec plusieurs profondeurs de racine.
