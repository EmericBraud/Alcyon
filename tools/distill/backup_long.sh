#!/bin/bash
# Tourne SUR LE SERVEUR GPU. Copie le reentrainement long vers gsrv:Alcyon-serveur/long
# (jeton Drive limite a drive.file : rclone ne voit que les fichiers qu'il a crees).
# Reseaux exportes, logs, pgn toutes les 30 min ; dernier checkpoint (1,8 Go) toutes les 3 h.
cd /workspace
D=gsrv:Alcyon-serveur/long
i=0
while true; do
  rclone copy nets_long $D/nets --include "e*/nnue/v5.nnue" 2>>backup.err
  rclone copy nets_anneal $D/nets_anneal --include "*/nnue/v5.nnue" 2>>backup.err
  rclone copy . $D --include "anneal*.status" --include "anneal_*.log" --include "anneal_*.pgn" --include "anneal_probe.sh" --include "sf8*" --max-depth 1 2>>backup.err
  rclone copy . $D --include "long.log" --include "long.status" --include "long_games.status" --include "long_val.status" --include "val_long.sh" \
    --include "long_e*.pgn" --include "long_e*.log" --include "run_long.sh" --include "watch_long.sh" \
    --include "backup_long.sh" --max-depth 1 2>>backup.err
  [ $((i % 6)) -eq 0 ] && [ -f long_last.ckpt ] && rclone copy long_last.ckpt $D 2>>backup.err
  echo "$(date +%H:%M:%S) sauvegarde $i" >> backup.log
  i=$((i + 1))
  sleep 1800
done
