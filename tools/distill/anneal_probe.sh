#!/bin/bash
# Sondes de recuit a partir du checkpoint e159 du reentrainement long (lr 9e-5, lambda ~0,90).
# A : lambda fixe 1,0 ; B : lambda fixe 0,90. Recuit : 6 epoques de 100 M, lr 9e-5 * 0,7^k
# (~1,1e-5 a la fin). Chaque reseau final joue 800 parties contre v5 (10+0.1), le match de A
# tourne pendant l'entrainement de B. Budget GPU total : < 2 h.
cd /workspace
. /workspace/venv/bin/activate
NP=/workspace/alcyon/training/nnue-pytorch
ST=/workspace/anneal.status
F="Full_Threats+HalfKAv2_hm^"
B=/workspace/v62/build/alcyon
D0=/workspace/v62/build/data
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
TR=$(ls /workspace/data/*.binpack | grep -v feb | tr "\n" " ")
[ -f /workspace/e159.pt ] || (cd $NP && python serialize.py --features "$F" /workspace/long_last.ckpt /workspace/e159.pt) || { echo "conversion e159 ratee" >> $ST; exit 1; }
match() {  # $1 = nom
  N=/workspace/nets_anneal/$1
  ./fastchess/fastchess -engine cmd=$N/run.sh name=$1 -engine cmd=/workspace/run_v5.sh name=v5 \
    -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
    -openings file=UHO_4060_v2.epd format=epd order=random -rounds 400 -games 2 -repeat \
    -concurrency 24 -recover -pgnout file=anneal_$1.pgn > anneal_$1.log 2>&1
  echo "$(date +%H:%M:%S) $1 : $(grep -E '^Elo' anneal_$1.log | tail -1) | $(grep -E '^Games' anneal_$1.log | tail -1) | pertes au temps $(grep -c 'time forfeit' anneal_$1.pgn)" >> $ST
}
for spec in "A 1.0" "B 0.90"; do
  set -- $spec; name=$1; lam=$2
  echo "$(date +%H:%M:%S) debut recuit $name (lambda $lam)" >> $ST
  (cd $NP && python train.py $TR --validation-datasets /workspace/ds/val_feb.binpack --validation-size 300000 \
    --features "$F" --resume-from-model /workspace/e159.pt --optimizer-name ranger21 \
    --lr 9e-5 --gamma 0.7 --lambda $lam --num-workers 24 --batch-size 16384 \
    --epoch-size 100000000 --max-epochs 6 --network-save-period 6 --save-top-k -1 \
    --default-root-dir /workspace/anneal_$name > /workspace/anneal_$name.train.log 2>&1)
  ck=$(ls -t /workspace/anneal_$name/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
  N=/workspace/nets_anneal/$name
  mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue
  (cd $NP && python serialize.py --features "$F" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { echo "export $name rate" >> $ST; continue; }
  printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
  echo "$(date +%H:%M:%S) recuit $name fini : $(tr '\r' '\n' < /workspace/anneal_$name.train.log | grep -E 'Epoch +5 \(Val\): \[val_loss' | tail -1)" >> $ST
  rm -f $ck
  match $name &
done
wait
echo "fini" >> $ST
