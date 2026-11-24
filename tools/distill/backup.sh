#!/bin/bash
# Sauvegarde periodique : instance AWS -> Mac -> Google Drive (gdrive:Alcyon).
# Seuls les morceaux termines (.plain.gz / .diag.gz) sont copies ; les .tmp en cours sont ignores.
H=ec2-user@13.53.45.176
K=$HOME/.ssh/mac-pro-chess.pem
L=/Users/emericbraud/Desktop/code/perso/Alcyon/gpu-archive
while true; do
  echo "== $(date '+%F %T')"
  rsync -a --exclude '*.tmp' -e "ssh -i $K -o ConnectTimeout=20" $H:gen/ $L/aws-gen/ && echo "rsync ok"
  rsync -a -e "ssh -i $K -o ConnectTimeout=20" $H:'*.py' $L/scripts/aws/ 2>/dev/null
  # dataset construit : binpacks et .plain compresses (les .plain bruts sont trop gros)
  rsync -a --exclude '*.plain' --exclude '*.tmp' -e "ssh -i $K -o ConnectTimeout=20" $H:ds/ $L/aws-ds/ 2>/dev/null && echo "rsync ds ok"
  python3 $L/pack.py
  rclone copy $L gdrive:Alcyon/gpu-archive --exclude 'aws-gen/**' --exclude '*.tmp' --transfers 4 --tpslimit 4 --log-level ERROR 2>/tmp/rclone_err.log && echo "rclone ok" || echo "rclone ECHEC : $(grep -o "reason\": \"[A-Z_]*" /tmp/rclone_err.log | sort -u | head -1)"
  echo "$(ls $L/aws-gen/run1/*.plain.gz 2>/dev/null | wc -l) morceaux, $(du -sh $L/aws-gen | cut -f1)"
  sleep 900
done
