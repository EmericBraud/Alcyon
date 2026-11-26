#!/bin/bash
# Refait fevrier 2024 (filtrage interrompu par erreur), en parallele de decembre et octobre.
ST=/workspace/more.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
cd /workspace/raw; f=test80-2024-02-feb-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack
log "telechargement $f (reprise)"
aria2c -q -x 16 -s 16 -k 64M --file-allocation=none -o $f https://huggingface.co/datasets/xushawn/test80-bt4-relabel/resolve/main/$f && \
  /workspace/condnet/filt_typ2 $f /workspace/bt4c2/${f%.binpack}.typ.binpack -1 5000000 2>> $ST && touch /workspace/raw/feb.ok
rm -f $f
until grep -q "PLUS_OK" $ST; do sleep 10; done
[ -f /workspace/raw/feb.ok ] && log "PLUS2_OK $(du -sh /workspace/bt4c2 | cut -f1) libre $(df -h / | tail -1 | awk '{print $4}')" || log "ECHEC fevrier"
