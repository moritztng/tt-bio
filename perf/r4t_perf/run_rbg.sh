#!/bin/bash
# r4t-perf R1: step-1 pair_update gradients, stock trimul caches (G0) vs train_in_proj (G1). Forward is bit-identical (step-1 loss equal).
cd ~/.coworker/wt/r4t-perf || exit 1
O=~/r4t/perf/rb; S=$HOME/r4t/step; PY=~/scratch/shipflow/venv-c4f99e695d34/bin/python
R(){ env PYTHONPATH=$PWD TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:r4t-perf TT_BIO_SDPA_OWN_FORWARD=1 timeout 400 $PY perf/r4t_step/trunk_step.py --n 256 --k 4 --batch $S/batch08.npz --exempt perf/r4t_step/g2_exempt.tsv --steps 1 "$@"; }
R4T_DUMP_G=$O/g0.npz R --out $O/ck_g0 > $O/G0.log 2>&1; echo "G0 RC $? $(date -u +%FT%TZ)" >> $O/run.log
R4T_DUMP_G=$O/g1.npz R4T_TRAIN_IN_PROJ=1 R --out $O/ck_g1 > $O/G1.log 2>&1; echo "G1 RC $? $(date -u +%FT%TZ)" >> $O/run.log
echo G-DONE >> $O/run.log
