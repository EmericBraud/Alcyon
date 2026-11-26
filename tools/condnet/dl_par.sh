#!/bin/bash
# Fin de la preparation en parallele : decembre et octobre 2023 telecharges et filtres en meme temps
# que le filtrage de fevrier 2024 (deja en cours, pid 21204).
ST=/workspace/more.status; log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
cd /workspace/raw
V=https://huggingface.co/datasets/vondele/linrock_relabel_2/resolve/main
for f in test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack; do
  ( log "telechargement $f (parallele)"; aria2c -q -x 16 -s 16 -k 64M --file-allocation=none -o $f $V/$f && /workspace/condnet/filt_typ2 $f /workspace/bt4c2/${f%.binpack}.typ.binpack -1 0 2>> $ST; rm -f $f ) &
done
while kill -0 21204 2>/dev/null; do sleep 10; done
rm -f /workspace/raw/test80-2024-02-feb-2tb7p.min-v2.v6.relabel-BT4-tf13tune-histfix.binpack
wait
log "PLUS_OK $(du -sh /workspace/bt4c2 | cut -f1) libre $(df -h / | tail -1 | awk '{print $4}')"
