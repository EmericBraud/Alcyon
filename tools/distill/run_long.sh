#!/bin/bash
# Reentrainement long de v5 (recette de reentrainement SF) sur les binpacks Leela filtres
# test80 min-v2.v6. lr 2e-4 decroissant (gamma 0,995 par epoque de 100 M), lambda 1,0 -> 0,75.
# Attend la fin du telechargement (dl.log : TOUT_FINI). Checkpoint toutes les 10 epoques.
cd /workspace/alcyon/training/nnue-pytorch
. /workspace/venv/bin/activate
until grep -q TOUT_FINI /workspace/dl.log; do sleep 30; done
D=/workspace/data
TR=$(ls $D/*.binpack | grep -v feb | tr "\n" " ")
echo "$(date +%H:%M:%S) donnees : $TR" >> /workspace/long.status
python train.py $TR --validation-datasets /workspace/ds/val_feb.binpack --validation-size 300000 \
  --features "Full_Threats+HalfKAv2_hm^" --resume-from-model /workspace/nets/v5.pt \
  --optimizer-name ranger21 --lr 2e-4 --gamma 0.995 --start-lambda 1.0 --end-lambda 0.75 \
  --num-workers 24 --batch-size 16384 --epoch-size 100000000 --max-epochs 400 --max-time 00:43:00:00 \
  --network-save-period 10 --save-top-k -1 --default-root-dir /workspace/long > /workspace/long.log 2>&1
echo "$(date +%H:%M:%S) entrainement termine rc $?" >> /workspace/long.status
