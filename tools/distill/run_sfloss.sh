#!/bin/bash
# Affinage du reseau BT4 avec la fonction de perte de la recette SF (vondele/nettest threats.yaml) :
# echelles et decalages de la sigmoide, exposant, asymetrie, ponderation par nombre de pieces et par
# ply, saut des 18 premiers plies. Memes donnees BT4 (test80 2024 mars a juin). 16 epoques,
# lr 1,5e-4 x 0,858 par epoque (fin ~1,5e-5), lambda 1,0.
# Puis analyse d'erreur (scores BT4), rapport eval/cible sur les avantages nets, match contre le reseau BT4.
cd /workspace; . venv/bin/activate
ST=/workspace/sfloss.status; E=/workspace/erran_bt4
NP=/workspace/alcyon/training/nnue-pytorch; F="Full_Threats+HalfKAv2_hm^"
log(){ echo "$(date +%H:%M:%S) $*" >> $ST; }
[ -f /workspace/bt4.pt ] || (cd $NP && python serialize.py --features "$F" /workspace/bt4_last.ckpt /workspace/bt4.pt > /dev/null 2>&1) || { log "conversion bt4 ratee"; exit 1; }
TR=$(ls /workspace/bt4/*.binpack | tr "\n" " ")
SFLOSS="--in-scaling 295.6539508488627 --out-scaling 379.98724077106635 --in-offset 285.2706341467852 --out-offset 289.1258344152218 \
 --pow-exp 2.4340402395048404 --qp-asymmetry 0.22866886710086187 \
 --pc-y0 -0.20 --pc-y1 0.45 --pc-y2 1.0 --pc-y3 0.95 --pc-y4 0.75 \
 --ply-x1 0.0 --ply-y1 0.025 --ply-x2 22.0 --ply-y2 0.05 --ply-x3 25.5 --ply-y3 0.20 --ply-x4 29.5 --ply-y4 0.80 \
 --early-fen-skipping 18 --random-fen-skipping 3"
log "debut entrainement (perte SF)"
(cd $NP && python train.py $TR --validation-datasets /workspace/ds/val_feb_bt4.binpack --validation-size 300000 \
  --features "$F" --resume-from-model /workspace/bt4.pt --optimizer-name ranger21 $SFLOSS \
  --lr 1.5e-4 --gamma 0.858 --lambda 1.0 --num-workers 24 --batch-size 16384 \
  --epoch-size 100000000 --max-epochs 16 --network-save-period 16 --save-top-k -1 \
  --default-root-dir /workspace/sfloss_run > /workspace/sfloss.train.log 2>&1)
ck=$(ls -t /workspace/sfloss_run/lightning_logs/version_*/checkpoints/*.ckpt | head -1)
[ -n "$ck" ] || { log "pas de checkpoint (voir sfloss.train.log)"; exit 1; }
D0=/workspace/v62/build/data; N=/workspace/nets_sfloss; B=/workspace/v62/build/alcyon
DESC=$(head -c 400 $D0/nnue/v5.nnue | tail -c +13 | head -c 240)
mkdir -p $N && cp -r $D0/. $N/ && rm -f $N/nnue/v2.nnue $N/nnue/v3.nnue
(cd $NP && python serialize.py --features "$F" --ft-compression none --description "$DESC" "$ck" $N/nnue/v5.nnue > /dev/null 2>&1) || { log "export rate"; exit 1; }
printf '#!/bin/sh\nexport ALCYON_DATA_DIR=%s\nexec %s\n' "$N" "$B" > $N/run.sh && chmod +x $N/run.sh
mv $ck /workspace/sfloss_last.ckpt
log "entrainement fini"
cd $E && cp evals_bt4.tsv evals_base4.tsv && python ev.py sfloss && python an.py sfloss > an_sfloss.out && log "analyse faite : $(sed -n 2p an_sfloss.out)"
python - >> $ST <<PY
import numpy as np
d=np.loadtxt("$E/evals_sfloss.tsv",delimiter="\t",usecols=(1,5,6,7))
sc=d[:,0]; sig=lambda x:1/(1+np.exp(-x/400))
m=(np.abs(sc)>=150)&(np.abs(sc)<400); s=np.sign(sc[m])
for j,n in [(1,"A"),(2,"sf14"),(3,"sfloss")]:
    e=d[:,j]; k=min(np.arange(0.5,2.01,0.02),key=lambda k:np.mean((sig(k*e)-sig(sc))**2))
    print(f"   {n}: echelle {k:.2f}, rapport eval/cible median sur 150-400 : {np.median(k*e[m]*s/(sc[m]*s)):.3f}")
PY
rclone copy /workspace/nets_sfloss/nnue gsrv:Alcyon-serveur/sfloss/nnue; rclone copy /workspace/sfloss.train.log gsrv:Alcyon-serveur/sfloss; rclone copy $E gsrv:Alcyon-serveur/sfloss --include "an_sfloss.out"
cd /workspace && ./fastchess/fastchess -engine cmd=$N/run.sh name=sfloss -engine cmd=/workspace/nets_bt4/run.sh name=bt4 \
  -each tc=10+0.1 proto=uci option.Threads=1 option.Hash=16 option.Ponder=false \
  -openings file=UHO_4060_v2.epd format=epd order=random -rounds 400 -games 2 -repeat \
  -concurrency 30 -recover -pgnout file=sfloss_vs_bt4.pgn > sfloss_vs_bt4.log 2>&1
log "sfloss vs bt4 : $(grep -E '^Elo' sfloss_vs_bt4.log | tail -1) | $(grep -E '^Games' sfloss_vs_bt4.log | tail -1) | pertes au temps $(grep -c 'time forfeit' sfloss_vs_bt4.pgn)"
rclone copy . gsrv:Alcyon-serveur/sfloss --include "sfloss*" --include "sfloss.status" --max-depth 1
log "fini"
