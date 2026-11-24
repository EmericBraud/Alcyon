#!/bin/bash
# Exporte les epoques choisies du fine-tuning de distillation et les oppose a v5
# (meme binaire v6.2), 10+0.1, 400 parties chacune, en serie. Usage : test_distill.sh 0 2 5
cd /workspace
. /workspace/venv/bin/activate
B=/workspace/v62/build/alcyon
D0=/workspace/v62/build/data
ST=/workspace/distill_games.status
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$D0" "$B" > /workspace/run_v5.sh && chmod +x /workspace/run_v5.sh
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
for e in "$@"; do
  ck=$(ls /workspace/ftd/lightning_logs/version_0/checkpoints/*epoch=$e-*.ckpt | head -1)
  N=/workspace/nets_d/e$e
  mkdir -p $N && cp -r $D0/. $N/
  (cd /workspace/alcyon/training/nnue-pytorch && python serialize.py --features "Full_Threats+HalfKAv2_hm^" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1)
  printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
  echo "$(date +%H:%M:%S) e$e bench $(printf 'bench\nquit\n' | $N/run.sh | tail -1)" >> $ST
done
for e in "$@"; do
  ./fastchess/fastchess -engine cmd=/workspace/nets_d/e$e/run.sh name=e$e -engine cmd=/workspace/run_v5.sh name=v5 \
    -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
    -openings file=UHO_4060_v2.epd format=epd order=random -rounds 200 -games 2 -repeat \
    -concurrency 30 -recover -pgnout file=dist_e$e.pgn > dist_e$e.log 2>&1
  echo "$(date +%H:%M:%S) e$e : $(grep -E '^Elo' dist_e$e.log | tail -1) | $(grep -E '^Games' dist_e$e.log | tail -1) | pertes au temps $(grep -c 'time forfeit' dist_e$e.pgn)" >> $ST
done
echo "fini" >> $ST
