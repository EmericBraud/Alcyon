#!/bin/sh
# Dump de l'etape 1 (docs/learned-pruning.md) sur tous les coeurs.
# Lots de 20 positions dans une file : chaque coeur libre prend le suivant,
# pour ne pas laisser des coeurs inactifs derriere un morceau lent.
set -e
rm -rf ~/run && mkdir -p ~/run && cd ~/run
n=$(wc -l < ~/fens_big.txt); k=$((n * 3 / 4))
head -n $k ~/fens_big.txt | split -l 20 -d -a 4 - tr_
tail -n +$((k + 1)) ~/fens_big.txt | split -l 20 -d -a 4 - te_
ls tr_???? te_???? | xargs -P "$(nproc)" -I{} sh -c \
  'printf "bench 16 $HOME/run/{}\nquit\n" | ALCYON_PRUNE_DUMP=$HOME/run/{}.bin ALCYON_PRUNE_DUMP_EVERY=1024 ~/alcyon/build-exp/alcyon > $HOME/run/{}.log 2>&1'
# Ajouter puis supprimer morceau par morceau : un cat global suivi d'un rm
# doublerait l'espace disque (23 Go de dump a bench 16 sur 57k positions).
for f in tr_????.bin; do cat "$f" >> train.bin && rm "$f"; done
for f in te_????.bin; do cat "$f" >> test.bin && rm "$f"; done
echo RUN_DONE
