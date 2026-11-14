#!/bin/bash
# Runs ON a live PNR box after box_pnr.sh: BindCraft 2 attn-stock rerun alone on the GPU (pnr-orchestrator 11-03).
#   bash box_bc2_rerun.sh <warm>      output /root/kc/bc2/rerun-attn-stock/{runs.jsonl,stdout.log,stderr.log,rc}
# The screen's attn-stock reading on B200 was a compile artifact: its cold trajectory stopped at 50 iterations, so the
# 140-iteration path compiled inside the first warm one. An empty PNR-DONE directory holds the session open meanwhile:
# box_pnr.sh's `echo done > PNR-DONE` fails on it and pnr_session's `ls PNR-DONE` prints nothing, so no destroy.
K=/root/kc; cd $K || exit 1
[ -f PNR-DONE ] && { echo "box_pnr already finished; too late to hold the session" >&2; exit 1; }
mkdir -p PNR-DONE
say(){ echo "$(date -u +%FT%TZ) bc2_rerun: $*" | tee -a box_pnr.log; }
say "holding PNR-DONE until box_pnr.sh exits"
while pgrep -f box_pnr.sh > /dev/null; do sleep 5; done
D=$K/bc2/rerun-attn-stock; mkdir -p $D; ln -sf ../../smi.csv $D/smi.csv
say "attn-stock --warm $1 alone on the GPU"
/opt/bc2-venv/bin/python $K/bc2_gpu.py --bc2 /root/bc2 --out $D --seed 101 --warm $1 --binder-len 80 \
  --set attention_backend=stock > $D/stdout.log 2> $D/stderr.log
echo $? > $D/rc; echo "bc2-rerun $(cat $D/rc) $(date -u +%FT%TZ)" >> PNR-PROGRESS
rmdir PNR-DONE; echo done > PNR-DONE; say "rerun rc=$(cat $D/rc); PNR-DONE"
