#!/bin/bash
# Environ toutes les 30 a 60 min (construction ~10-20 min + pause 30 min) : reconstruit le dataset (binpack) a partir de toute la generation,
# et arrete la generation des que l'objectif de positions utiles est atteint.
TARGET=${1:-20000000}
cd ~
while true; do
  rm -rf ~/ds/next
  nice python3 build_ds.py --out ~/ds/next --keep-every 3 > ~/ds_next.log 2>&1
  for p in ~/ds/next/part*.plain; do nice ~/sftools/src/stockfish convert $p ${p%.plain}.binpack > /dev/null 2>&1 && rm -f $p; done
  N=$(grep -o "total [0-9]*" ~/ds_next.log | awk '{print $2}')
  rm -rf ~/ds/latest && mv ~/ds/next ~/ds/latest && cp ~/ds_next.log ~/ds/latest/composition.log
  echo "$(date '+%F %T') dataset : $N positions" >> ~/auto_ds.status
  if [ "${N:-0}" -ge "$TARGET" ]; then
    tmux kill-session -t gen
    echo "$(date '+%F %T') objectif $TARGET atteint, generation arretee" >> ~/auto_ds.status
    break
  fi
  sleep 1800
done
