#!/bin/bash
# Sonde "redemarrage a chaud" : A repris avec un lr eleve (5e-4, 2,5 x le debut du run long)
# puis decroissance geometrique jusqu'a ~1,5e-5 en 22 epoques (gamma 0,846), lambda 1,0, donnees
# d'origine (test80 2023-11, 2024 mars a juin). Question : le plateau vient-il du creux herite de v5 ?
# Match contre A seulement si la val loss finale passe sous 0,00290 (A : 0,00295).
cd /workspace; . venv/bin/activate
ST=/workspace/hilr.status; E=/workspace/erran
NP=/workspace/alcyon/training/nnue-pytorch; F="Full_Threats+HalfKAv2_hm^"
log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
TR=$(ls /workspace/data/*.binpack | grep -v feb | tr "\n" " ")
log "debut entrainement : $TR"
(cd $NP && python train.py $TR --validation-datasets /workspace/ds/val_feb.binpack --validation-size 300000 \
  --features "$F" --resume-from-model /workspace/A.pt --optimizer-name ranger21 \
  --lr 5e-4 --gamma 0.846 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 22 --network-save-period 22 --save-top-k -1 \
  --default-root-dir /workspace/hilr_run > /workspace/hilr.train.log 2>&1)
ck=$(ls -t /workspace/hilr_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
D0=/workspace/v62/build/data; N=/workspace/nets_hilr; B=/workspace/v62/build/alcyon
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue
(cd $NP && python serialize.py --features "$F" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { log "export rate"; exit 1; }
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
mv $ck /workspace/hilr_last.ckpt
VL=$(tr '\r' '\n' < /workspace/hilr.train.log | grep -oE 'Epoch +21 \(Val\): \[val_loss_epoch=[0-9.]+' | grep -oE '0\.[0-9]+' | tail -1)
log "entrainement fini : val_loss $VL"
cd $E && python ev.py hilr && python an.py hilr > an_hilr.out && log "analyse faite (an_hilr.out) : $(sed -n 2p an_hilr.out)"
rclone copy /workspace/nets_hilr/nnue gsrv:Alcyon-serveur/hilr/nnue; rclone copy /workspace/hilr.train.log gsrv:Alcyon-serveur/hilr; rclone copy $E/an_hilr.out gsrv:Alcyon-serveur/hilr
if python -c "import sys; sys.exit(0 if float('$VL') < 0.00290 else 1)"; then
  cd /workspace && ./fastchess/fastchess -engine cmd=$N/run.sh name=hilr -engine cmd=/workspace/nets_anneal/A/run.sh name=A \
    -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
    -openings file=UHO_4060_v2.epd format=epd order=random -rounds 400 -games 2 -repeat \
    -concurrency 30 -recover -pgnout file=hilr_vs_A.pgn > hilr_vs_A.log 2>&1
  log "hilr vs A : $(grep -E '^Elo' hilr_vs_A.log | tail -1) | $(grep -E '^Games' hilr_vs_A.log | tail -1)"
  rclone copy . gsrv:Alcyon-serveur/hilr --include "hilr_vs_A*" --max-depth 1
else
  log "val loss pas sous 0,00290 : pas de match"
fi
log "fini"
