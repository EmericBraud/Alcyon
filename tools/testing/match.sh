#!/bin/sh
# Match a nombre fixe de parties, meme binaire, une variante (options UCI) contre la base.
#   match.sh <binaire> <nom> [option=valeur ...]
# Ex. : match.sh /data/alcyon/build-nott/alcyon ph1 persist_history=1
# Base = memes options communes (BASE_OPTS), sans les options de la variante.
# 4000 parties en 5+0.05, Hash 8, 1 thread ; tri rapide des variantes avant un SPRT.
BIN=$1; N=$2; shift 2
BASE_OPTS=${BASE_OPTS:-"option.learned_prune_enabled=0"}
CONC=${CONC:-180}
opts=""
for o in "$@"; do opts="$opts option.$o"; done
export ALCYON_DATA_DIR=${ALCYON_DATA_DIR:-/data/alcyon/data}
ulimit -n 65536
timeout 5000 /data/Client/fastchess-ob -recover \
  -engine cmd=$BIN name=$N $BASE_OPTS $opts \
  -engine cmd=$BIN name=base $BASE_OPTS \
  -each tc=5+0.05 option.Threads=1 option.Hash=8 \
  -openings file=/data/Client/Books/UHO_4060_v2.epd format=epd order=random \
  -rounds 2000 -games 2 -repeat -concurrency $CONC > /data/match_$N.log 2>&1
echo "== $N : illegal $(grep -ci "illegal move" /data/match_$N.log), incidents $(grep -ciE "disconnects|loses on time" /data/match_$N.log)"
grep -A4 "^Results of" /data/match_$N.log | tail -5
