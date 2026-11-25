#!/bin/bash
# Fine-tuning de A sur un melange 75 % positions d'origine (test80 juin 2024) / 20 % positions
# desequilibrees ou de finale (test80 2023 juillet et octobre filtres par filt.cpp) / 5 % hse-v1
# (dual-nnue, gros desequilibres). Proportions en positions, le chargeur tirant les fichiers au
# prorata de leur taille en octets (2,6 / 9 / 3,4 octets par position).
cd /workspace; . venv/bin/activate
ST=/workspace/mixdes.status; E=/workspace/erran; MX=/workspace/mixdes; mkdir -p $MX
NP=/workspace/alcyon/training/nnue-pytorch; F="Full_Threats+HalfKAv2_hm^"
HF=https://huggingface.co/datasets/linrock
log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
for m in 07-jul 10-oct; do
  o=$MX/f2023-$m.binpack; [ -f $o.done ] && continue
  log "filtrage 2023-$m"
  rm -f $MX/ff && mkfifo $MX/ff
  (curl -sL $HF/test80-2023/resolve/main/test80-2023-$m-2tb7p.min-v2.v6.binpack.zst | zstd -dc > $MX/ff 2>/dev/null &)
  $E/filt $MX/ff $o 480000000 2>> $ST && touch $o.done; rm -f $MX/ff
done
[ -f $MX/hse2.binpack ] || curl -sL $HF/dual-nnue/resolve/main/hse-v1/test80-aug2023-2tb7p.v6.min.high-simple-eval-1k.min-v2.binpack -o $MX/hse2.binpack
cp -n $E/hse.binpack $MX/hse1.binpack
log "donnees : $(ls -la $MX/*.binpack /workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack | awk '{print $5, $9}' | tr '\n' ' ')"
TR="/workspace/data/test80-2024-06-jun-2tb7p.min-v2.v6.binpack $MX/f2023-07-jul.binpack $MX/f2023-10-oct.binpack $MX/hse1.binpack $MX/hse2.binpack"
log "debut entrainement"
(cd $NP && python train.py $TR --validation-datasets /workspace/ds/val_feb.binpack --validation-size 300000 \
  --features "$F" --resume-from-model /workspace/A.pt --optimizer-name ranger21 \
  --lr 6e-5 --gamma 0.85 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 10 --network-save-period 10 --save-top-k -1 \
  --default-root-dir /workspace/mixdes_run > /workspace/mixdes.train.log 2>&1)
ck=$(ls -t /workspace/mixdes_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
D0=/workspace/v62/build/data; N=/workspace/nets_mixdes; B=/workspace/v62/build/alcyon
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue
(cd $NP && python serialize.py --features "$F" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { log "export rate"; exit 1; }
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
cp $ck /workspace/mixdes_last.ckpt && rm -f $ck
log "entrainement fini : $(tr '\r' '\n' < /workspace/mixdes.train.log | grep -E 'Epoch +9 \(Val\): \[val_loss' | tail -1)"
rclone copy $N/nnue gsrv:Alcyon-serveur/mixdes/nnue; rclone copy /workspace/mixdes.train.log gsrv:Alcyon-serveur/mixdes
cd $E && python ev.py mixdes && python an.py mixdes > an_mixdes.out && log "analyse faite (an_mixdes.out)"
rclone copy $E/an_mixdes.out gsrv:Alcyon-serveur/mixdes
cd /workspace && ./fastchess/fastchess -engine cmd=$N/run.sh name=mixdes -engine cmd=/workspace/nets_anneal/A/run.sh name=A \
  -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
  -openings file=UHO_4060_v2.epd format=epd order=random -rounds 400 -games 2 -repeat \
  -concurrency 30 -recover -pgnout file=mixdes_vs_A.pgn > mixdes_vs_A.log 2>&1
log "mixdes vs A : $(grep -E '^Elo' mixdes_vs_A.log | tail -1) | $(grep -E '^Games' mixdes_vs_A.log | tail -1) | pertes au temps $(grep -c 'time forfeit' mixdes_vs_A.pgn)"
rclone copy . gsrv:Alcyon-serveur/mixdes --include "mixdes*" --max-depth 1
log "fini"
