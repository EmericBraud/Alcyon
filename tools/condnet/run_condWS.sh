#!/bin/bash
# W (bloc 128) typ entraine 24 epoques (lr 5e-4 x 0,885) : v7 fige (accumulateur et piles), piles de couches propres aux positions sous condition (copies entrainables), bloc de 128 (condition "typ" : pieces differentes), entraine
# sur les seules positions sous condition de test80 2024 mars-juin reevalues par BT4.
# lr 5e-4 x 0,774 par epoque sur 12 epoques (fin ~3e-5), lambda 1,0. Puis evals de v7 et du bloc
# sur la validation de fevrier BT4 et analyse d'erreur par type de position.
cd /workspace; . venv/bin/activate
ST=/workspace/condWS.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
F="Full_Threats+HalfKAv2_hm^"
cd np && python /workspace/condnet/patch_np.py /workspace/condnet/cond_model.py >> $ST 2>&1
[ -f /workspace/nets/v7.pt ] || python serialize.py --features "$F" /workspace/nets/v7.nnue /workspace/nets/v7.pt >> $ST 2>&1
PYTHONPATH=. python /workspace/condnet/make_cond.py /workspace/nets/v7.pt /workspace/nets/condWS_init.pt 128 typ W >> $ST 2>&1
[ -f /workspace/ds/val_feb_bt4.tsv ] || /workspace/dump /workspace/ds/val_feb_bt4.binpack > /workspace/ds/val_feb_bt4.tsv
[ -f /workspace/ds/val_feb_bt4_typ.binpack ] || /workspace/condnet/filt_typ /workspace/ds/val_feb_bt4.binpack /workspace/ds/val_feb_bt4_typ.binpack 2>/dev/null
PYTHONPATH=. python /workspace/condnet/check_cond.py /workspace/nets/v7.pt /workspace/nets/condWS_init.pt /workspace/ds/val_feb_bt4.tsv >> $ST 2>&1 || { log "verification ECHEC"; exit 1; }
TR=$(ls /workspace/bt4c/*.typ.binpack | tr "\n" " ")
log "debut phase 1 : $TR"
python train.py $TR --validation-datasets /workspace/ds/val_feb_bt4_typ.binpack --validation-size 250000 \
  --features "$F" --resume-from-model /workspace/nets/condWS_init.pt --optimizer-name ranger21 \
  --lr 5e-4 --gamma 0.885 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 24 --network-save-period 8 --save-top-k -1 \
  --default-root-dir /workspace/condWS_run > /workspace/condWS.train.log 2>&1
ck=$(ls -t /workspace/condWS_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
[ -n "$ck" ] || { log "pas de checkpoint (voir condWS.train.log)"; exit 1; }
log "entrainement fini : $ck"
PYTHONPATH=. python /workspace/condnet/eval_fens.py /workspace/ds/val_feb_bt4.tsv /workspace/condWS_evals.tsv v7=/workspace/nets/v7.pt "W128=/workspace/nets/condW_init.pt@/workspace/condW_run/lightning_logs/version_0/checkpoints/last.ckpt" "W128L=/workspace/nets/condWS_init.pt@$ck" "W512L=/workspace/nets/condWL_init.pt@/workspace/condWL_run/lightning_logs/version_0/checkpoints/last.ckpt" >> $ST 2>&1
python /workspace/condnet/an2.py /workspace/condWS_evals.tsv v7 W128 W128L W512L > /workspace/condWS_an.out 2>&1 && log "analyse faite : $(sed -n 2p /workspace/condWS_an.out)"
log "fini"
