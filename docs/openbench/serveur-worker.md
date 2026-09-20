# Monter un serveur worker OpenBench

Procedure pour brancher une instance EC2 neuve sur l'OpenBench local et y
faire tourner des matchs chess26. Ecrite apres plusieurs mises en place ou
les memes pieges ont coute des heures : chacun est signale par **PIEGE**.

## Le modele

L'OpenBench tourne **en local sur le Mac** (Django, port 8000), expose via
ngrok. Le serveur EC2 ne fait que tourner des parties : il demande du travail,
compile les deux moteurs depuis GitHub, joue, renvoie les resultats. Il ne
stocke rien d'important — une instance qu'on eteint ne perd que du temps de
compilation.

Consequence : **les commits a tester doivent etre pousses sur GitHub** avant
de creer le test, et l'URL de l'archive est figee a la creation du test (elle
utilise le SHA de l'arbre). Un correctif pousse apres coup ne sera jamais
telecharge — il faut recreer le test.

## 1. Acces SSH

La cle `chess26_tuning.pem` doit etre presente et en mode 600. Creer un
script d'acces plutot que de retaper les options :

```sh
#!/bin/sh
SP=<dossier de travail>
exec ssh -i "$SP/chess26_tuning.pem" -o UserKnownHostsFile="$SP/known_hosts" \
     -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
     -o BatchMode=yes -o ServerAliveInterval=30 ubuntu@<IP> "$@"
```

`BatchMode=yes` evite de rester bloque sur une demande de passphrase, et
`ConnectTimeout` evite d'attendre 2 minutes quand l'instance est eteinte.

## 2. Dependances

Les AMI Ubuntu serveur n'ont ni compilateur ni cmake.

```sh
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
     build-essential cmake git-lfs \
     python3-requests python3-psutil python3-cpuinfo
```

**PIEGE** — `py-cpuinfo` manquant fait echouer `import worker` dans le client,
et le message affiche est trompeur :

```
Exception: Client missing, and --no-client-downloads provided
```

Ce n'est pas le client qui manque, c'est une de ses dependances. Lire
`Client/requirements.txt` et tout installer d'un coup au lieu de decouvrir les
imports un par un. Diagnostic direct :

```sh
cd ~/Client && python3 -c "import worker"
```

## 3. Client OpenBench et livre d'ouvertures

Transferer l'archive du client (elle contient `Books/`, donc le livre) :

```sh
scp -i cle.pem client.tgz ubuntu@<IP>:~/
ssh ... 'tar xzf client.tgz && cp Client/Books/UHO_4060_v2.epd ~/'
```

**PIEGE** — sans livre d'ouvertures, fastchess sort a la premiere ligne du log
et joue **zero partie**. Le symptome est un test qui reste a 0 partie alors
que le worker semble tourner. Verifier :

```sh
wc -l < ~/UHO_4060_v2.epd    # doit donner 242201
```

## 4. Donnees du moteur : le piege principal

chess26 cherche son dossier `data/` **a cote de son executable**. OpenBench ne
copie que le binaire dans `Client/Engines/`. Sans rien faire, tout moteur
compile meurt sur :

```
FATAL: Magics file couln't be opened for read operation
```

Et les reseaux NNUE sont en **Git LFS** : un `git clone` nu ne rapatrie que des
pointeurs de 134 octets.

```sh
git clone -q --depth 1 -b <branche> https://github.com/EmericBraud/chess26 ~/sweep
cd ~/sweep && git lfs install && git lfs pull
ls -la ~/sweep/data/nnue/v3.nnue     # doit faire 111262528 octets
```

Puis designer ce dossier par variable d'environnement — plus robuste qu'un
lien symbolique dans `Client/Engines/`, qu'un nettoyage d'OpenBench effacerait :

```sh
export CHESS26_DATA_DIR=$HOME/sweep/data
```

Verification avant de lancer quoi que ce soit :

```sh
CHESS26_DATA_DIR=$HOME/sweep/data ~/Client/Engines/<Binaire> bench | tail -1
```

Le nombre de noeuds doit etre **exactement** celui du bench local du meme
commit. S'il differe, le binaire distant n'explore pas le meme arbre et toute
mesure sera fausse.

## 5. Lancer le worker

```sh
#!/bin/sh
# Une partie = 4 tuyaux. A 164 parties simultanees la limite Ubuntu par
# defaut (1024) est depassee, d'ou "pipe() failed" dans fastchess.
ulimit -n 65536
export CHESS26_DATA_DIR=$HOME/sweep/data
cd /home/ubuntu/Client
exec python3 -u client.py -U <user> -P <pass> -S <url ngrok> \
     -T 164 -N 1 --no-client-downloads
```

Sous tmux pour survivre a la deconnexion SSH :

```sh
tmux new-session -d -s wk "sh ~/launch164.sh 2>&1 | tee -a ~/worker.log"
```

`-T` est la **concurrency**, pas un nombre de workers. Un seul processus
client suffit. Sur une machine a 192 vCPU, 164 laisse de la marge pour le
systeme. Ne pas confondre avec les lignes `client.py` vues par `ps` : le
wrapper tmux, le `bash -c` et le forkserver multiprocessing donnent 4 lignes
pour **un seul** worker.

## 6. Verifier — toujours, avant de rapporter quoi que ce soit

```sh
pgrep -fc Chess26      # ~2x la concurrency (un processus par camp)
uptime                 # load ~= concurrency
grep -c "illegal\|FATAL\|Warning\|disconnect" ~/worker.log
tail -3 ~/worker.log   # des lignes "Finished game"
```

**PIEGE** — un test cree **avant** que l'environnement soit pret est marque
`finished=1, games=0` par OpenBench et ne redemarrera jamais. Il faut le
recreer. Donc : monter le serveur d'abord, verifier le bench, puis creer le
test.

**PIEGE** — ne jamais conclure sur la foi d'une commande lancee : lire le log.
Plusieurs fois, un "match lance" s'est revele etre zero partie jouee, avec
l'erreur en ligne 1 du log.

## 7. Cote Mac

- `caffeinate` sur le processus Django si l'ecran peut se verrouiller.
- L'URL ngrok change a chaque redemarrage du tunnel ; le lanceur du serveur
  la contient en dur, donc la resynchroniser.
- Depuis le serveur : `curl -s -o /dev/null -w "%{http_code}" <url>/index/`
  doit rendre 200.

## Memo des symptomes

| Symptome | Cause |
|---|---|
| `Client missing, and --no-client-downloads` | dependance Python manquante (souvent `py-cpuinfo`) |
| `FATAL: Magics file couln't be opened` | `CHESS26_DATA_DIR` absent |
| Reseau NNUE de 134 octets | `git lfs pull` non fait |
| Test a 0 partie, worker actif | livre d'ouvertures absent |
| Test `finished=1, games=0` | cree avant que le bench fonctionne — recreer |
| `pipe() failed` | `ulimit -n` trop bas |
| Bench distant != bench local | mauvais commit ou mauvaises donnees |
