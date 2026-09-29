#!/bin/sh
# Dump de docs/learned-pruning.md sur tous les coeurs.
#   [ALCYON_PRUNE_DUMP_LABEL=1] sh run_dump.sh <binaire> <positions> <dossier de sortie> <EVERY> [profondeur]
# Lots de 20 positions dans une file : chaque coeur libre prend le suivant,
# pour ne pas laisser des coeurs inactifs derriere un morceau lent.
# Mecanisme appris force a 0 : les donnees viennent de l'arbre sans lui.
set -e
BIN=$1; FENS=$2; OUT=$3; EVERY=$4; DEPTH=${5:-16}
export BIN OUT EVERY DEPTH
rm -rf "$OUT" && mkdir -p "$OUT" && cd "$OUT"
n=$(wc -l < "$FENS"); k=$((n * 3 / 4))
head -n $k "$FENS" | split -l 20 -d -a 4 - tr_
tail -n +$((k + 1)) "$FENS" | split -l 20 -d -a 4 - te_
ls tr_???? te_???? | xargs -P "$(nproc)" -I{} sh -c \
  '{ echo "setoption name learned_prune_enabled value 0"; echo "bench $DEPTH $OUT/{}"; echo quit; } | ALCYON_PRUNE_DUMP=$OUT/{}.bin ALCYON_PRUNE_DUMP_EVERY=$EVERY "$BIN" > $OUT/{}.log 2>&1'
# Ajouter puis supprimer morceau par morceau : un cat global suivi d'un rm
# doublerait l'espace disque.
for p in tr te; do
  for f in ${p}_????.bin; do
    cat "$f" >> $p.bin && rm "$f"
    [ -f "$f.l0" ] && cat "$f.l0" >> $p.bin.l0 && rm "$f.l0"
    [ -f "$f.v3" ] && touch $p.bin.v3 && rm "$f.v3"
  done
done
echo RUN_DONE
