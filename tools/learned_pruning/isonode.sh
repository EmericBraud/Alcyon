#!/bin/sh
# Base a noeuds egaux, position par position (docs/learned-pruning.md).
#   sh isonode.sh <binaire SPSA> <dossier de sweep.sh> <config> [config...]
# Pour chaque config, relit ses logs (noeuds par position), ecrit un fichier
# de budgets par lot, et relance le moteur SANS le mecanisme avec ces
# budgets exacts (bench nodesfile). Sorties : iso_<config>__lot_XXXX.log.
set -e
BIN=$1; OUT=$2; shift 2
export BIN OUT
mkdir -p "$OUT/budgets"
for cfg in "$@"; do
  for log in "$OUT"/${cfg}__lot_*.log; do
    lot=$(basename "$log" .log | sed "s/^${cfg}__//")
    grep "info string bench [0-9]*/" "$log" | sed -E 's/.* nodes ([0-9]+) .*/\1/' \
      | paste -d' ' - "$OUT/lots/$lot" > "$OUT/budgets/${cfg}__$lot"
    echo "$cfg $lot"
  done
done | xargs -P "$(nproc)" -L1 sh -c \
  '{ echo "setoption name learned_prune_enabled value 0"; echo "bench nodesfile $OUT/budgets/$0__$1"; echo quit; } | "$BIN" > "$OUT/iso_$0__$1.log" 2>&1'
echo ISO_DONE
