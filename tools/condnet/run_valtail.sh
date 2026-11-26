#!/bin/bash
# Evalue tous les reseaux deja entraines (aucun reentrainement) sur la validation tiree de la fin
# jamais utilisee des mois d entrainement (ds/val_tail), puis analyse d erreur par type.
cd /workspace; . venv/bin/activate
until grep -qE " fini$|ECHEC|pas de checkpoint" /workspace/condWS.status; do sleep 20; done
R=/workspace; N=$R/nets; L=lightning_logs/version_0/checkpoints/last.ckpt
cd np && PYTHONPATH=. python /workspace/condnet/eval_fens.py $R/ds/val_tail.tsv $R/valtail_evals.tsv \
  v7=$N/v7.pt A=$N/cond_init.pt@$R/cond1_run/$L B=$N/condB_init.pt@$R/condB_run/$L W=$N/condW_init.pt@$R/condW_run/$L \
  W512=$N/condW5_init.pt@$R/condW5_run/$L WL=$N/condWL_init.pt@$R/condWL_run/$L WS=$N/condWS_init.pt@$R/condWS_run/$L \
  D=$N/condD_init.pt@$R/condD_run/$L > $R/valtail.log 2>&1
python /workspace/condnet/an2.py $R/valtail_evals.tsv v7 A B W W512 WS WL D > $R/valtail_an.out 2>&1
echo "$(date +%H:%M:%S) fini" >> $R/valtail.log
