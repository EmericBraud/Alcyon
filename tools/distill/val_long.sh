#!/bin/bash
# Tourne sur le serveur : extrait la val loss de chaque epoque du reentrainement long dans
# /workspace/long_val.status (epoque, lr theorique, val loss), mis a jour toutes les 5 min.
while true; do
  tr '\r' '\n' < /workspace/long.log | grep -E "^Epoch +[0-9]+ \(Val\): \[val_loss" | sort -u -k2,2n | \
    awk '{gsub(/[^0-9.]/,"",$4); printf "epoque %3d  lr %.2e  val_loss %s\n", $2, 2e-4*0.995^$2, $4}' > /workspace/long_val.status.tmp
  mv /workspace/long_val.status.tmp /workspace/long_val.status
  sleep 300
done
