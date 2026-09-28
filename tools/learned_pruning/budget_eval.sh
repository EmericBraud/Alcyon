#!/bin/sh
# Evaluation SYMETRIQUE a budget egal (docs/learned-pruning.md).
#   sh budget_eval.sh <binaire SPSA> <dossier de sweep.sh> <fichier de configs> <config> [config...]
# Chaque position recoit un budget commun B = noeuds(off12) x U, U dans
# [1, 1.8] tire par hachage de (lot, index) : ni pile sur une fin
# d'iteration, ni plus loin qu'off13. La base (learned_prune_enabled=0) et
# chaque config cherchent avec CE budget (bench nodesfile), meme binaire :
# chacun perd son iteration inachevee de la meme facon. Une option
# "budget_scale=X" dans la config multiplie son budget (X = nps_config /
# nps_base : comparaison a temps egal). Sorties : nb_<config>__lot_XXXX.log,
# base nb_base__lot_XXXX.log.
set -e
BIN=$1; OUT=$2; CONFIGS=$3; shift 3
export BIN OUT
mkdir -p "$OUT/nbudgets"
for log in "$OUT"/off12__lot_*.log; do
  lot=$(basename "$log" .log | sed 's/^off12__//'); n=$(echo "${lot#lot_}" | sed 's/^0*//'); n=${n:-0}
  grep "info string bench [0-9]*/" "$log" | sed -E 's/.* nodes ([0-9]+) .*/\1/' \
    | awk -v lot="$n" '{ u = 1 + 0.8 * (((NR * 2654435761 + lot * 40503) % 1000) / 1000); printf "%d\n", $1 * u }' \
    | paste -d' ' - "$OUT/lots/$lot" > "$OUT/nbudgets/$lot"
done
{
  for lot in $(ls "$OUT/nbudgets"); do echo "base|1|learned_prune_enabled=0|$lot"; done
  for cfg in "$@"; do
    opts=$(grep "^$cfg|" "$CONFIGS" | cut -d'|' -f3)
    scale=$(echo "$opts" | tr ' ' '\n' | sed -n 's/^budget_scale=//p'); scale=${scale:-1}
    opts=$(echo "$opts" | tr ' ' '\n' | grep -v '^budget_scale=' | tr '\n' ' ')
    for lot in $(ls "$OUT/nbudgets"); do echo "$cfg|$scale|$opts|$lot"; done
  done
} | xargs -P "$(nproc)" -d '\n' -I{} sh -c '
  IFS="|" read -r name scale opts lot <<JOB
{}
JOB
  awk -v k="$scale" "{ \$1 = int(\$1 * k); print }" "$OUT/nbudgets/$lot" > "$OUT/nbudgets/.${name}_$lot"
  { for o in $opts; do echo "setoption name ${o%%=*} value ${o#*=}"; done
    echo "bench nodesfile $OUT/nbudgets/.${name}_$lot"; echo quit; } | "$BIN" > "$OUT/nb_${name}__$lot.log" 2>&1'
echo BUDGET_DONE
