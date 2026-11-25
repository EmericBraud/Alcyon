#!/bin/bash
# Mesure contre Stockfish 8 (protocole du README : 60+0.2, Threads 1, Hash 64, UHO_4060_v2),
# meme machine pour le nouveau reseau et pour v5 (reference v6.2), en serie, concurrence 30.
# Usage : sf8_gauntlet.sh <run.sh du nouveau reseau> <nom> [rounds]
cd /workspace
NEW=$1; NAME=$2; R=${3:-200}
ST=/workspace/sf8.status
for spec in "$NEW $NAME" "/workspace/run_v5.sh v5"; do
  set -- $spec
  ./fastchess/fastchess -engine cmd=$1 name=$2 -engine cmd=/workspace/sf8-src/src/stockfish name=sf8 \
    -each tc=60+0.2 proto=uci option.Threads=1 option.Hash=64 option.Ponder=false \
    -openings file=UHO_4060_v2.epd format=epd order=random -rounds $R -games 2 -repeat \
    -concurrency 30 -recover -pgnout file=sf8_$2.pgn > sf8_$2.log 2>&1
  echo "$(date +%H:%M:%S) $2 vs SF8 : $(grep -E '^Elo' sf8_$2.log | tail -1) | $(grep -E '^Games' sf8_$2.log | tail -1) | pertes au temps $(grep -c 'time forfeit' sf8_$2.pgn)" >> $ST
done
echo "fini" >> $ST
