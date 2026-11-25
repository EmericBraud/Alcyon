#!/bin/bash
# Surveille /workspace/long : chaque nouveau checkpoint (toutes les 10 epoques) est exporte en
# .nnue, le .ckpt est supprime (disque), puis match de 400 parties contre v5 (10+0.1, meme binaire v6.2).
cd /workspace
. /workspace/venv/bin/activate
B=/workspace/v62/build/alcyon
D0=/workspace/v62/build/data
ST=/workspace/long_games.status
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$D0" "$B" > /workspace/run_v5.sh && chmod +x /workspace/run_v5.sh
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
while true; do
  for ck in $(ls /workspace/long/lightning_logs/version_*/checkpoints/epoch=*.ckpt 2>/dev/null); do
    e=$(basename $ck | sed 's/epoch=\([0-9]*\)-.*/\1/')
    N=/workspace/nets_long/e$e
    [ -f $N/nnue/v5.nnue ] && { rm -f $ck; continue; }
    sleep 20   # laisser finir l'ecriture
    mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue   # 2 x 111 Mo inutiles
    (cd /workspace/alcyon/training/nnue-pytorch && python serialize.py --features "Full_Threats+HalfKAv2_hm^" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { echo "export e$e rate" >> $ST; continue; }
    cp $ck /workspace/long_last.ckpt.tmp && mv /workspace/long_last.ckpt.tmp /workspace/long_last.ckpt && rm -f $ck
    printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
    ./fastchess/fastchess -engine cmd=$N/run.sh name=e$e -engine cmd=/workspace/run_v5.sh name=v5 \
      -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
      -openings file=UHO_4060_v2.epd format=epd order=random -rounds 200 -games 2 -repeat \
      -concurrency 24 -recover -pgnout file=long_e$e.pgn > long_e$e.log 2>&1
    echo "$(date +%H:%M:%S) e$e : $(grep -E '^Elo' long_e$e.log | tail -1) | $(grep -E '^Games' long_e$e.log | tail -1) | pertes au temps $(grep -c 'time forfeit' long_e$e.pgn) | val_loss $(tr '\r' '\n' < /workspace/long.log | grep -E "Epoch +$e \(Val\): \[val_loss" | grep -oE '[0-9]\.[0-9]+' | tail -1)" >> $ST
  done
  grep -q "entrainement termine" /workspace/long.status 2>/dev/null && ! ls /workspace/long/lightning_logs/version_*/checkpoints/epoch=*.ckpt >/dev/null 2>&1 && break
  sleep 60
done
echo "fini" >> $ST
