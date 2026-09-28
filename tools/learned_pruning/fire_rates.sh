#!/bin/sh
# Taux de declenchement du mecanisme par config (docs/learned-pruning.md).
#   sh fire_rates.sh <binaire SPSA+experiences> <positions> <configs> <profondeur>
# Une ligne par config : "<nom> <declenchements> <noeuds> <taux>".
BIN=$1; FENS=$2; CONFIGS=$3; DEPTH=$4
export BIN FENS DEPTH
grep -v '^#' "$CONFIGS" | xargs -P "$(nproc)" -d '\n' -I{} sh -c '
  name=$(echo "{}" | cut -d"|" -f1); opts=$(echo "{}" | cut -d"|" -f3)
  { for o in $opts; do echo "setoption name ${o%%=*} value ${o#*=}"; done
    echo "bench $DEPTH $FENS"; echo prunestats; echo quit; } | ALCYON_PRUNE_STATS=1 "$BIN" 2>&1 \
  | awk -v name="$name" "
      /prunestats depth=/ { for (i = 1; i <= NF; i++) { if (\$i ~ /^n=/) n = substr(\$i, 3); if (\$i ~ /^learned_fires=/) { f = substr(\$i, 15); sub(/%/, \"\", f); fires += n * f / 100 } } }
      /^[0-9]+ nodes/ { nodes = \$1 }
      END { printf \"%s %d %d %.4f\\n\", name, fires, nodes, fires / nodes }"'
