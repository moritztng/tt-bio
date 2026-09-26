#!/bin/bash
cd /home/ttuser/.coworker/wt/bcx-p10-trimove || exit 1
WT=$PWD
mkdir -p "$WT/perf/bcx_p10_trimove/out/round_p1"
export TRIMOVE_AB_ARMS="${ARMS:-off,on}"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-p10-trimove
export PYTHONPATH="$WT"
exec /home/ttuser/bcx_e2e_venv/bin/python -u "$WT/perf/bcx_p10_trimove/round_ab.py" \
  --rounds 9 --seed 100 --exact 0 --shipped --binder 146 \
  --out "$WT/perf/bcx_p10_trimove/out/${OUTDIR:-round_p1}"
