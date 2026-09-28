#!/bin/sh
# Base a noeuds egaux, position par position (docs/learned-pruning.md).
#   [SCALE=1.07] [PREFIX=iso] sh isonode.sh <binaire SPSA> <dossier de sweep.sh> <config> [config...]
# SCALE multiplie le budget de la base : pour un mecanisme qui coute du nps,
# SCALE = nps_base / nps_config rend la comparaison a TEMPS egal.
# Pour chaque config, relit ses logs (noeuds par position), ecrit un fichier
# de budgets par lot, et relance le moteur SANS le mecanisme avec ces
# budgets exacts (bench nodesfile). Sorties : iso_<config>__lot_XXXX.log.
set -e
BIN=$1; OUT=$2; shift 2
SCALE=${SCALE:-1}; PREFIX=${PREFIX:-iso}
export BIN OUT PREFIX
mkdir -p "$OUT/budgets"
for cfg in "$@"; do
  for log in "$OUT"/${cfg}__lot_*.log; do
    lot=$(basename "$log" .log | sed "s/^${cfg}__//")
    grep "info string bench [0-9]*/" "$log" | sed -E 's/.* nodes ([0-9]+) .*/\1/' \
      | awk -v k="$SCALE" '{ printf "%d\n", $1 * k }' \
      | paste -d' ' - "$OUT/lots/$lot" > "$OUT/budgets/${PREFIX}_${cfg}__$lot"
    echo "$cfg $lot"
  done
done | xargs -P "$(nproc)" -L1 sh -c \
  '{ echo "setoption name learned_prune_enabled value 0"; echo "bench nodesfile $OUT/budgets/${PREFIX}_$0__$1"; echo quit; } | "$BIN" > "$OUT/${PREFIX}_$0__$1.log" 2>&1'
echo ISO_DONE
