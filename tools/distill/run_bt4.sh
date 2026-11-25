#!/bin/bash
# Affinage de A sur les memes mois test80 (2024 mars a juin), reevalues par Leela BT4
# (xushawn/test80-bt4-relabel, versions -histfix). lr 3e-4 puis x0,854 par epoque sur 20
# epoques (fin ~1,5e-5), lambda 1,0. Validation : 300 000 positions de fevrier 2024 BT4.
# Puis analyse d'erreur par type (scores BT4) et match de 800 parties contre A si l'erreur
# globale baisse d'au moins 0,3 (x1e4) par rapport a A.
cd /workspace; . venv/bin/activate
ST=/workspace/bt4run.status; E=/workspace/erran_bt4
NP=/workspace/alcyon/training/nnue-pytorch; F="Full_Threats+HalfKAv2_hm^"
log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
until grep -q TOUT_FINI /workspace/bt4.status 2>/dev/null; do sleep 30; done
TR=$(ls /workspace/bt4/*.binpack | tr "\n" " ")
log "debut entrainement : $TR"
(cd $NP && python train.py $TR --validation-datasets /workspace/ds/val_feb_bt4.binpack --validation-size 300000 \
  --features "$F" --resume-from-model /workspace/A.pt --optimizer-name ranger21 \
  --lr 3e-4 --gamma 0.854 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 20 --network-save-period 20 --save-top-k -1 \
  --default-root-dir /workspace/bt4_run > /workspace/bt4run.train.log 2>&1)
ck=$(ls -t /workspace/bt4_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
D0=/workspace/v62/build/data; N=/workspace/nets_bt4; B=/workspace/v62/build/alcyon
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue
(cd $NP && python serialize.py --features "$F" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { log "export rate"; exit 1; }
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
mv $ck /workspace/bt4_last.ckpt
log "entrainement fini : $(tr '\r' '\n' < /workspace/bt4run.train.log | grep -oE 'Epoch +19 \(Val\): \[val_loss_epoch=[0-9.]+' | tail -1)"
cd $E && python ev.py bt4 && python an.py bt4 > an_bt4.out && log "analyse faite : $(sed -n 2p an_bt4.out)"
rclone copy /workspace/nets_bt4/nnue gsrv:Alcyon-serveur/bt4/nnue; rclone copy /workspace/bt4run.train.log gsrv:Alcyon-serveur/bt4; rclone copy $E gsrv:Alcyon-serveur/bt4 --include "an_*.out" --include "*.py"
if python - <<PY
import re,sys
l=open("$E/an_bt4.out").read().splitlines()[1]
v=dict(re.findall(r"(\w+) ([0-9.]+)",l.split("erreur (x1e4)")[1]))
sys.exit(0 if float(v["A"])-float(v["bt4"])>=0.3 else 1)
PY
then
  cd /workspace && ./fastchess/fastchess -engine cmd=$N/run.sh name=bt4 -engine cmd=/workspace/nets_anneal/A/run.sh name=A \
    -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
    -openings file=UHO_4060_v2.epd format=epd order=random -rounds 400 -games 2 -repeat \
    -concurrency 30 -recover -pgnout file=bt4_vs_A.pgn > bt4_vs_A.log 2>&1
  log "bt4 vs A : $(grep -E '^Elo' bt4_vs_A.log | tail -1) | $(grep -E '^Games' bt4_vs_A.log | tail -1) | pertes au temps $(grep -c 'time forfeit' bt4_vs_A.pgn)"
  rclone copy . gsrv:Alcyon-serveur/bt4 --include "bt4_vs_A*" --max-depth 1
else
  log "erreur pas assez en baisse : pas de match"
fi
log "fini"
