#!/bin/bash
# r4t-perf R1 timing: interleaved A (lazy-forward rebuild, R4T_REBUILD_INLINE=0) / B (inline+prewarm, default) x2, 5 steps each.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/rbab; mkdir -p $O; S=$HOME/r4t/step; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d" " -f1-3 /proc/loadavg)" > $O/run.log
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 600 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv --steps 5 "$@"; }
for r in 1 2; do
 R4T_REBUILD_INLINE=0 R --out $O/ck_a$r > $O/A$r.log 2>&1; echo "A$r RC $? $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)" >> $O/run.log
 R --out $O/ck_b$r > $O/B$r.log 2>&1; echo "B$r RC $? $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)" >> $O/run.log
done
echo AB-DONE >> $O/run.log
