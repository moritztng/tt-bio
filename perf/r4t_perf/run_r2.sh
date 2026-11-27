#!/bin/bash
# r4t-perf R2: grade (a) + broken arm (d), then interleaved A (host pack) / B (R4T_DEVICE_PACK=1) x2, 5 steps each.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/r2; mkdir -p $O; S=$HOME/r4t/step; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
B="--n 256 --k 4 --batch $S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv"
E(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 900 "$@"; }
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD) load $(cut -d" " -f1-3 /proc/loadavg)" > $O/run.log
E $PY perf/r4t_perf/r2grade.py $O/grade.json perf/r4t_step/trunk_step.py $B --steps 1 --out $O/ck_g > $O/grade.log 2>&1; echo "GRADE RC $? $(date -u +%FT%TZ)" >> $O/run.log
E $PY perf/r4t_perf/r2grade.py --broken $O/grade_broken.json perf/r4t_step/trunk_step.py $B --steps 1 --out $O/ck_gb > $O/grade_broken.log 2>&1; echo "BROKEN RC $? $(date -u +%FT%TZ)" >> $O/run.log
for r in 1 2; do
 E $PY perf/r4t_step/trunk_step.py $B --steps 5 --out $O/ck_a$r > $O/A$r.log 2>&1; echo "A$r RC $? $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)" >> $O/run.log
 R4T_DEVICE_PACK=1 E $PY perf/r4t_step/trunk_step.py $B --steps 5 --out $O/ck_b$r > $O/B$r.log 2>&1; echo "B$r RC $? $(date -u +%FT%TZ) load $(cut -d" " -f1-3 /proc/loadavg)" >> $O/run.log
done
echo R2-DONE >> $O/run.log
