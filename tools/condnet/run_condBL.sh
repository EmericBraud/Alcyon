#!/bin/bash
# Variante B (bloc 128, piles propres de la taille de v7 : 32-32, condition typ) entrainee longtemps :
# 48 epoques sur les 4,8 milliards de positions sous condition (test80 2024 mars-juin + 2023 oct, dec,
# 2024 jan, fev), lr 5e-4 x 0,942 par epoque (fin ~3e-5). Puis evaluation en quantification simulee
# de v7, B (12 epoques) et BL sur les deux validations.
cd /workspace; . venv/bin/activate
ST=/workspace/condBL.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
R=/workspace; N=$R/nets; L=lightning_logs/version_0/checkpoints/last.ckpt
cd np && cp /workspace/condnet/cond_model.py model/cond_model.py
PYTHONPATH=. python /workspace/condnet/make_cond.py $N/v7.pt $N/condBL_init.pt 128 typ B >> $ST 2>&1
PYTHONPATH=. python /workspace/condnet/check_cond.py $N/v7.pt $N/condBL_init.pt $R/ds/val_feb_bt4.tsv >> $ST 2>&1 || { log "verification ECHEC"; exit 1; }
TR=$(ls $R/bt4c/*.typ.binpack $R/bt4c2/*.typ.binpack | tr "\n" " ")
log "debut : $TR"
python train.py $TR --validation-datasets $R/ds/val_feb_bt4_typ.binpack --validation-size 250000 \
  --features "Full_Threats+HalfKAv2_hm^" --resume-from-model $N/condBL_init.pt --optimizer-name ranger21 \
  --lr 5e-4 --gamma 0.942 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 48 --network-save-period 12 --save-top-k -1 \
  --default-root-dir $R/condBL_run > $R/condBL.train.log 2>&1
ck=$(ls -t $R/condBL_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
[ -n "$ck" ] || { log "pas de checkpoint"; exit 1; }
log "entrainement fini : $ck"
PYTHONPATH=. python -c "
import torch
n = torch.load('$N/condBL_init.pt', weights_only=False, map_location='cpu')
n.load_state_dict(torch.load('$ck', weights_only=False, map_location='cpu')['state_dict'])
torch.save(n, '$N/BL.pt'); print('BL.pt ecrit')
" >> $ST 2>&1
for v in tail feb_bt4; do
  PYTHONPATH=. python /workspace/condnet/eval_fens_fq.py $R/ds/val_$v.tsv $R/bl_$v.tsv v7=$N/v7.pt \
    B=$N/condB_init.pt@$R/condB_run/$L BL=$N/BL.pt >> $ST 2>&1
  python /workspace/condnet/an2.py $R/bl_$v.tsv v7 B BL > $R/bl_${v}_an.out 2>&1
  log "analyse $v (quantif simulee) : $(sed -n 2p $R/bl_${v}_an.out)"
done
log "fini"
