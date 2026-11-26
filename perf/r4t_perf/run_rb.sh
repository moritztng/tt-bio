#!/bin/bash
# r4t-perf R1: P = train_in_proj with the per-rebuild lazy forward kept; I = train_in_proj + inline recording. Bar: I digests == P digests every step.
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/rb; S=$HOME/r4t/step; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
echo "START $(date -u +%FT%TZ) sha $(git rev-parse --short HEAD)" > $O/run.log
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 600 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $S/batch08.npz,$S/batch01.npz,$S/batch05.npz,$S/batch00.npz --exempt perf/r4t_step/g2_exempt.tsv --steps 5 "$@"; }
[ -n "$ONLY_I" ] || { R4T_TRAIN_IN_PROJ=1 R --out $O/ck_p > $O/P.log 2>&1; echo "P RC $? $(date -u +%FT%TZ)" >> $O/run.log; }
R4T_REBUILD_INLINE=1 R --out $O/ck_i > $O/I${ISFX}.log 2>&1; echo "I RC $? $(date -u +%FT%TZ)" >> $O/run.log
echo RB-DONE >> $O/run.log
