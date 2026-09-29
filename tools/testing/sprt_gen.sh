#!/bin/sh
# SPRT de gain [0, 3] : sprt_gen.sh <branche> <binaire de reference> <nom>
BR=$1; REF=$2; N=$3
cd /data/alcyon && git fetch -q origin $BR
[ -d /data/sprt_$N ] || git worktree add -f /data/sprt_$N origin/$BR > /dev/null 2>&1
(cd /data/sprt_$N && git checkout -q --detach origin/$BR && git log --oneline -1 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_NNUE_EVAL=ON > /data/sprt_$N.cfg.log 2>&1 && cmake --build build -j100 > /data/sprt_$N.build.log 2>&1) || { echo BUILD_FAIL; tail -20 /data/sprt_$N.build.log; exit 1; }
export ALCYON_DATA_DIR=/data/alcyon/data
ulimit -n 65536
for b in /data/sprt_$N/build/alcyon $REF; do printf 'bench\nquit\n' | $b 2>&1 | grep -E "^[0-9]+ nodes"; done
cd /data/sprt_$N
/data/Client/fastchess-ob -recover \
  -engine cmd=/data/sprt_$N/build/alcyon name=$N \
  -engine cmd=$REF name=main \
  -each tc=8+0.08 option.Threads=1 option.Hash=16 \
  -openings file=/data/Client/Books/UHO_4060_v2.epd format=epd order=random \
  -sprt elo0=0 elo1=3 alpha=0.05 beta=0.05 \
  -rounds 100000 -games 2 -repeat -concurrency 180 -ratinginterval 200 > /data/sprt_$N.log 2>&1
echo "illegal $(grep -ci "illegal move" /data/sprt_$N.log), deconnexions $(grep -ciE "disconnects|loses on time" /data/sprt_$N.log)"
grep -A6 "^Results of" /data/sprt_$N.log | tail -7
echo SPRT_DONE
