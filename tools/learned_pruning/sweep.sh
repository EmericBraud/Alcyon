#!/bin/sh
# Etape 3 de docs/learned-pruning.md : balayage hors parties, sur tous les coeurs.
#   sh sweep.sh <binaire SPSA> <fichier de positions> <sweep_configs.txt> <dossier de sortie>
# Chaque job = (config, lot de 25 positions) ; une file xargs occupe tous les coeurs.
set -e
BIN=$1; FENS=$2; CONFIGS=$3; OUT=$4
rm -rf "$OUT" && mkdir -p "$OUT/lots"
split -l 25 -d -a 4 "$FENS" "$OUT/lots/lot_"
grep -v '^#' "$CONFIGS" | while IFS='|' read -r name depth opts; do
  for lot in "$OUT"/lots/lot_*; do echo "$name|$depth|$opts|$lot"; done
done > "$OUT/jobs.txt"
export BIN OUT
# Le ref (profondeur 14) est de loin le plus long : trie en tete pour ne pas finir sur lui.
sort -t'|' -k2,2nr "$OUT/jobs.txt" | xargs -P "$(nproc)" -d '\n' -I{} sh -c '
  IFS="|" read -r name depth opts lot <<JOB
{}
JOB
  { for o in $opts; do echo "setoption name ${o%%=*} value ${o#*=}"; done
    echo "bench $depth $lot"; echo quit; } | "$BIN" > "$OUT/${name}__$(basename $lot).log" 2>&1'
echo SWEEP_DONE
