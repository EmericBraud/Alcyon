#!/bin/bash
# Fine-tuning de v5 sur le dataset de distillation seul (lambda 1 : scores SF uniquement,
# les resultats viennent de parties Alcyon-SF, biaises). Usage : run_distill.sh [smoke]
cd /workspace/alcyon/training/nnue-pytorch
. /workspace/venv/bin/activate
TR=/workspace/ds/train_distill.binpack
VD=/workspace/ds/val_distill.binpack
VF=/workspace/ds/val_feb.binpack
ST=/workspace/distill.status
C="--features Full_Threats+HalfKAv2_hm^ --lambda 1.0 --optimizer-name ranger21 --num-workers 16 --batch-size 16384"
ref() {  # $1 = validation, $2 = nom : perte de v5 (lr quasi nul)
  python train.py $TR $C --validation-datasets $1 --validation-size 300000 --resume-from-model /workspace/nets/v5.pt \
    --lr 1e-9 --epoch-size 100000 --max-epochs 1 --default-root-dir /workspace/ref_$2 > /workspace/ref_$2.log 2>&1
  echo "$(date +%H:%M:%S) v5 $2 $(grep -o 'val_loss_epoch=[0-9.]*' /workspace/ref_$2.log | tail -1)" >> $ST
}
if [ "$1" = smoke ]; then
  python train.py $TR $C --validation-datasets $VD --validation-size 100000 --resume-from-model /workspace/nets/v5.pt \
    --lr 3e-5 --epoch-size 200000 --max-epochs 1 --default-root-dir /workspace/smoke > /workspace/smoke.log 2>&1
  echo "smoke rc $? $(grep -o 'val_loss_epoch=[0-9.]*' /workspace/smoke.log | tail -1)" >> $ST
  exit
fi
ref $VD distill
ref $VF feb
echo "$(date +%H:%M:%S) fine-tuning" >> $ST
python train.py $TR $C --validation-datasets $VD --validation-size 300000 --resume-from-model /workspace/nets/v5.pt \
  --lr 3e-5 --gamma 0.8 --epoch-size 9000000 --max-epochs 6 --network-save-period 1 --save-top-k -1 \
  --default-root-dir /workspace/ftd > /workspace/ftd.log 2>&1
grep -o "Epoch *[0-9]* (Val): \[val_loss_epoch=[0-9.]*" /workspace/ftd.log >> $ST
echo "$(date +%H:%M:%S) fini" >> $ST
