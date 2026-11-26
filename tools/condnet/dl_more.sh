#!/bin/bash
# Telecharge, filtre (condition typ) et supprime un mois a la fois. Fevrier 2024 : 5 M premieres
# positions sautees (la validation de fevrier vient du debut du fichier).
ST=/workspace/more.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
mkdir -p /workspace/raw /workspace/bt4c2; cd /workspace/raw
X=https://huggingface.co/datasets/xushawn/test80-bt4-relabel/resolve/main
V=https://huggingface.co/datasets/vondele/linrock_relabel_2/resolve/main
for spec in "$X test80-2024-01-jan-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack 0" \
            "$X test80-2024-02-feb-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack 5000000" \
            "$V test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack 0" \
            "$V test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack 0"; do
  set -- $spec; base=$1; f=$2; skip=$3
  log "telechargement $f (libre $(df -h / | tail -1 | awk '{print $4}'))"
  aria2c -q -x 16 -s 16 -k 64M --file-allocation=none -o $f $base/$f || { log "ECHEC telechargement $f"; continue; }
  /workspace/condnet/filt_typ2 $f /workspace/bt4c2/${f%.binpack}.typ.binpack -1 $skip 2>> $ST
  rm -f $f
done
log "PLUS_OK $(du -sh /workspace/bt4c2 | cut -f1) libre $(df -h / | tail -1 | awk '{print $4}')"
