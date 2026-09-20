# Creer un tune SPSA dans OpenBench

Procedure et pieges, ecrits apres avoir cree quatre tunes successifs dont
**trois etaient inutilisables** — pour des raisons toutes verifiables en
trente secondes avant lancement.

## Avant toute chose : trois questions

Y repondre evite la quasi-totalite des tunes gaspilles.

### 1. Le parametre est-il reellement expose ?

La liste des options SPSA est **maintenue a la main** dans
`src/interface/uci.hpp` (bloc `#ifdef SPSA_TUNING`). Un `PARAM_SPECIFIER` qui
n'y figure pas est intunable, silencieusement.

```sh
printf 'uci\nquit\n' | ./chess26 | grep "option name <nom>"
```

Verifier aussi qu'il n'est pas **mort** : changer sa valeur doit changer
l'arbre.

```sh
printf 'setoption name <nom> value <autre>\nbench\nquit\n' | ./chess26 | tail -1
```

Si le compte de noeuds ne bouge pas, le parametre n'est pas lu sur le chemin
chaud. Cas vecu : `killer_score` et `counter_score` etaient des constantes
vestigiales — `MovePicker` ordonne par etapes (`GOOD_CAPTURES → KILLERS →
COUNTERS → QUIETS`), donc les killers passent avant les coups calmes par
construction, quelle que soit leur valeur. Deux dimensions mortes dans un
tune.

### 2. Des parties sont-elles vraiment necessaires ?

Pour un parametre de **pur ordonnancement a effet local**, la taille d'arbre a
profondeur fixe est un objectif deterministe, sans bruit, mesurable en
secondes au lieu de dizaines de milliers de parties. Voir la section
« Balayage par taille d'arbre ».

Ce n'est **pas** valide pour :

- un parametre de **reduction ou d'elagage** : moins de noeuds est trivialement
  atteignable en elaguant plus, l'objectif est donc faux ;
- un parametre a **accumulation longue** (bornes d'history) : l'effet vit sur
  un horizon plus long que le bench.

### 3. Les parametres pourront-ils bouger ?

C'est le piege le plus couteux. Voir la section suivante.

## La verification qui compte : la course

OpenBench precalcule a la creation (`OpenBench/spsa_utils.py`) :

```python
c_value = c_end * iterations ** gamma
a_value = (r_end * c_end ** 2) * (a_ratio * iterations + iterations) ** alpha
```

puis a chaque iteration :

```python
c = max(c_value / iteration ** gamma, 0.50 if int else 0.0)
r = a_value / (a_ratio * iterations + iteration) ** alpha / c ** 2
```

Le deplacement maximal par iteration vaut `r * c`. La **course totale** est sa
somme sur toutes les iterations. Elle doit etre du meme ordre que la plage du
parametre : viser un ratio de **0,8 a 1,5**.

Script a executer avant chaque creation :

```python
alpha, gamma, a_ratio, it = 0.602, 0.101, 0.1, <iterations>
def course(c_end, r_end):
    cv = c_end * it ** gamma
    av = r_end * c_end ** 2 * (a_ratio * it + it) ** alpha
    t = 0
    for i in range(1, it + 1):
        c = max(cv / i ** gamma, 0.5)
        t += av / ((a_ratio * it + i) ** alpha) / c ** 2 * c
    return t
for name, c_end, r_end, rng in PARAMS:
    print(f"{name:24} course={course(c_end,r_end):9.1f} ratio={course(c_end,r_end)/rng:5.2f}")
```

**PIEGE VECU** — un tune lance avec `r_end = 1.2e-6` sur des plages de 30 000 :
course de 17 unites, soit **0,05 % de la plage**. `killer_score` partait de
8000 et pouvait au mieux atteindre 8017. 96 000 parties auraient mesure un
seul des quatre parametres. La valeur venait d'une regle extrapolee de tunes
passes (`r_end = 0.04 / plage`) : numeriquement exacte sur son domaine
d'origine — de petites plages sur 100 000 iterations — et fausse d'un facteur
mille ailleurs. **Le wiki donne la valeur de reference : `r_end = 0.002`.**

## Hyperparametres

| champ | valeur | note |
|---|---|---|
| `alpha` | 0.602 | ne pas toucher |
| `gamma` | 0.101 | ne pas toucher |
| `a_ratio` | 0.1 | ne pas toucher |
| `pairs_per` | 8 | bon compromis |
| `reporting` | BATCHED | |
| `distribution` | **MULTIPLE** sur machine unique | voir la section dediee ; SINGLE n'a de sens qu'avec une flotte |
| `iterations` | voir ci-dessous | parties = `iterations * pairs_per * 2` |

`c_end` : environ **1/20e de la plage raisonnable**.

Dimensionner `iterations` par le nombre de parametres, pas par habitude :
2 parametres n'ont pas besoin de 96 000 parties. Compter quelques milliers de
parties par parametre, puis **verifier la course** — reduire les iterations la
reduit mecaniquement, et il faut alors remonter `r_end` ou resserrer les
plages.

**`c_value` et `a_value` sont figes a la creation.** Changer `iterations` ou
`pairs_per` impose de recreer le tune. En revanche `min_value`, `max_value` et
`scale_nps` sont modifiables a chaud.

## Ne jamais faire partir un parametre sur une borne

La mise a jour du serveur est :

```python
param.value = max(param.min_value, min(param.max_value, param.value + delta))
```

Sur une borne, les deltas qui pointent vers l'exterieur sont ecrases et ceux
qui pointent vers l'interieur sont conserves. **Sous bruit pur, sans aucun
gradient reel, l'esperance de la valeur derive donc vers l'interieur.** Un
parametre parti de 0 et fini a 0.4 ne prouve rien : c'est peut-etre la seule
trace du mur.

S'ajoute une perte de sensibilite. A `value == min`, la perturbation `-c` est
ramenee a la borne, donc les deux camps jouent `min` contre `min + c` : la
separation effective est `c` et non `2c`, alors que le pas applique reste
calibre pour `2c`.

Le signe, lui, reste correct : le delta ne depend que du `flip`, pas des
valeurs reellement jouees, et le flip encode bien quel camp portait la valeur
haute.

Donc faire partir chaque parametre **strictement a l'interieur** de sa plage.
Un atterrissage sur une borne devient alors un signal -- le parametre a du
parcourir la distance -- au lieu d'un artefact.

## Distribution : SINGLE ou MULTIPLE ?

Le wiki recommande SINGLE. **Cette recommandation suppose une flotte de
machines** : chaque machine tire son propre workload, donc N machines = N
points SPSA evalues en parallele.

Sur **une seule machine**, SINGLE donne un point a la fois, et la taille du
workload croissant avec la concurrency, ce point est evalue par des milliers
de parties au lieu des `2 * pairs_per` prevues. Mesure sur un tune a
concurrency 164 : **6 workloads pour 17 448 parties**, soit six directions de
gradient echantillonnees. Le compteur d'iterations du serveur, lui, avance
aux parties :

```python
iteration = 1 + games / (pairs_per * 2)
```

Il croyait donc etre a l'iteration 1024 sur 2000 et avait fait decroitre `c`
et `r` en consequence : le pas s'eteint pendant que l'exploration n'a pas eu
lieu.

En MULTIPLE, `permutations = runner_count` et chaque runner recoit son propre
point. Le client lance alors une copie du match runner par point, a
concurrency 2 : a concurrency 160, **80 points distincts en vol**. C'est le
bon reglage pour une machine unique.

Verification apres lancement :

```sh
grep -c "Workload \[" ~/worker.log     # doit croitre regulierement
```

## Parametres empoisonnes

Pour un entier, OpenBench force `c = max(..., 0.5)`. Sur une plage tres courte
(0..3), `c` est donc au plancher des les premieres iterations et le parametre
oscille entre ses bornes au lieu de converger. Le wiki appelle ca un
« parametre empoisonne ».

Pour ceux-la, **un balayage manuel sur trois valeurs repond plus vite qu'un
SPSA**. Reserver les parties aux vraies variables continues (marges, pentes).

## Format d'entree

Sept champs par ligne :

```
nom, int|float, valeur, min, max, c_end, r_end
```

```
rfp_marg_d_fact, int, 79, 29, 129, 5.0, 0.004
lmr_cut_node_bonus, int, 2, 0, 6, 0.3, 0.010
```

## Creation par HTTP

Django exige le jeton CSRF et l'en-tete `Referer`.

```python
import re, sys, requests
S = "http://127.0.0.1:8000"
s = requests.Session()
s.get(S + "/login/", timeout=10)
tok = s.cookies.get("csrftoken")
s.post(S + "/login/", data={"csrfmiddlewaretoken": tok, "username": U, "password": P},
       headers={"Referer": S + "/login/"}, timeout=10)
r = s.get(S + "/tune/new/", timeout=15)
tok = s.cookies.get("csrftoken") or re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
form = {
    "csrfmiddlewaretoken": tok,
    "dev_engine": "Chess26", "dev_repo": "https://github.com/EmericBraud/chess26",
    "dev_branch": "<SHA complet, pousse sur GitHub>", "dev_bench": "<bench exact>",
    "dev_options": "Threads=1 Hash=8", "dev_network": "",
    "dev_time_control": "10.0+0.10", "book_name": "UHO_4060_v2.epd",
    "spsa_reporting_type": "BATCHED", "spsa_distribution_type": "SINGLE",
    "spsa_alpha": "0.602", "spsa_gamma": "0.101", "spsa_A_ratio": "0.1",
    "spsa_iterations": "3000", "spsa_pairs_per": "8",
    "spsa_inputs": open(sys.argv[3]).read().strip(),
    "win_adj": "movecount=4 score=500",
    "draw_adj": "movenumber=32 movecount=6 score=6",
    "upload_pgns": "FALSE", "syzygy_wdl": "DISABLED", "syzygy_adj": "DISABLED",
    "scale_method": "DEV", "scale_nps": "<nps mesure>",
    "priority": "0", "throughput": "1000", "submit": "Create SPSA Tune",
}
r = s.post(S + "/tune/new/", data=form, headers={"Referer": S + "/tune/new/"}, timeout=60)
print("POST ->", r.status_code, "| url:", r.url)
```

Un POST accepte redirige vers `/index/`. Approuver ensuite :
`GET /test/<id>/APPROVE/` avec la meme session. **Ne pas editer
`db.sqlite3` directement** pour approuver.

## Apres lancement : verifier, une fois

```python
import sqlite3
c = sqlite3.connect('db.sqlite3'); c.row_factory = sqlite3.Row
rid = list(c.execute('select id from OpenBench_spsarun order by id desc limit 1'))[0][0]
for p in c.execute('select name,value,min_value,max_value,c_end,r_end '
                   'from OpenBench_spsaparameter where spsa_run_id=?', (rid,)):
    print(dict(p))
```

Les `c_end` / `r_end` doivent etre ceux voulus, et `value` doit s'ecarter de
son depart apres quelques centaines de parties. Une valeur parfaitement
immobile signale une course nulle.

## Balayage par taille d'arbre (alternative aux parties)

`bench <depth> <fichier>` accepte un fichier de positions (FEN ou EPD).

**PIEGE** — sur les 16 positions codees en dur, le compte de noeuds saute de
13 % d'une valeur a l'autre et l'optimum **change a chaque profondeur** :
l'ordonnancement se propage en cascade et rien ne moyenne cette cascade. C'est
deterministe mais chaotique, donc inexploitable. Il faut **plusieurs centaines
de positions** ; a 570 l'ecart tombe a 1,8 % et le classement devient stable.

Toujours confirmer a **deux profondeurs differentes** : un classement qui
s'inverse est du chaos, pas un signal.

Parallelisation (190 coeurs, un processus par lot de positions, ~1 minute) :

```sh
split -n l/190 -d -a 3 positions.epd $WORK/chunk
ls $WORK/chunk* | xargs -P 190 -I{} sh -c \
  "printf 'setoption name $OPT value $V\nbench $DEPTH {}\nquit\n' | $ENG 2>/dev/null \
   | tail -1 | awk '{print \$1}'" | awk '{s+=\$1} END {print s}'
```

Ecrire le resultat dans un fichier et le relire, plutot que de relancer le
balayage pour en extraire une autre colonne.

## Appliquer les resultats

Les valeurs finales s'appliquent dans **les deux blocs** de
`src/engine/config/config.hpp` (NNUE et HCE) — ils sont separes par
`#ifdef NNUE_EVAL`, et n'en modifier qu'un laisse l'autre en arriere.

Puis valider par SPRT. Un tune est optimise **en self-play** ; un gain en
self-play se transfere a un adversaire plus fort a environ la moitie de sa
valeur. La validation qui compte se fait **contre Stockfish 8**.
