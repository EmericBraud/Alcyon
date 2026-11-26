#!/bin/bash
# Suite de l'entrainement de WS (bloc 128, tete 128-64, condition typ) sur des donnees neuves :
# test80 2024 janvier et fevrier (histfix, debut de fevrier exclu), 2023 octobre et decembre (BT4),
# filtrees sur la condition. Repart des poids de WS (24 epoques), lr 2e-4 x 0,905 sur 24 epoques
# (fin ~2e-5). Puis evaluation de v7, WS et WS2 sur les deux validations (fin vierge de mars-juin,
# debut de fevrier).
cd /workspace; . venv/bin/activate
ST=/workspace/condWS2.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
until grep -q "PLUS_OK" /workspace/more.status 2>/dev/null; do sleep 30; done
R=/workspace; N=$R/nets; L=lightning_logs/version_0/checkpoints/last.ckpt
cd np && cp /workspace/condnet/cond_model.py model/cond_model.py
PYTHONPATH=. python -c "
import torch
n = torch.load('$N/condWS_init.pt', weights_only=False, map_location='cpu')
n.load_state_dict(torch.load('$R/condWS_run/$L', weights_only=False, map_location='cpu')['state_dict'])
torch.save(n, '$N/condWS_trained.pt'); print('poids de WS ecrits')
" >> $ST 2>&1
TR=$(ls /workspace/bt4c2/*.typ.binpack | tr "\n" " ")
log "debut : $TR"
python train.py $TR --validation-datasets /workspace/ds/val_feb_bt4_typ.binpack --validation-size 250000 \
  --features "Full_Threats+HalfKAv2_hm^" --resume-from-model $N/condWS_trained.pt --optimizer-name ranger21 \
  --lr 2e-4 --gamma 0.905 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 24 --network-save-period 8 --save-top-k -1 \
  --default-root-dir $R/condWS2_run > $R/condWS2.train.log 2>&1
ck=$(ls -t $R/condWS2_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
[ -n "$ck" ] || { log "pas de checkpoint"; exit 1; }
log "entrainement fini : $ck"
for v in tail feb_bt4; do
  PYTHONPATH=. python /workspace/condnet/eval_fens.py $R/ds/val_$v.tsv $R/ws2_$v.tsv v7=$N/v7.pt \
    WS=$N/condWS_init.pt@$R/condWS_run/$L "WS2=$N/condWS_init.pt@$ck" >> $ST 2>&1
  python /workspace/condnet/an2.py $R/ws2_$v.tsv v7 WS WS2 > $R/ws2_${v}_an.out 2>&1
  log "analyse $v : $(sed -n 2p $R/ws2_${v}_an.out)"
done
log "fini"
