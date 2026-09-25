#!/bin/bash
# bcx-tracewire: the fenced capture against the eager bisect leg, 2 design steps, extra-MSA on.
cd /home/ttuser/.coworker/wt/bcx-tracewire || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_tracewire/logs; S=$L/fence_test.status; : > $S
rm -rf perf/bcx_tracewire/runs/fence_trace_seed100
timeout 1200 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm trace --digest --extra-msa \
  --rounds 3 --seed 100 --card 3 --project perf/bcx_tracewire/runs/fence_trace_seed100 \
  --out fence_xmsa_trace_seed100.json > $L/fence_trace.log 2>&1 < /dev/null
echo "trace exit $? $(date -u +%FT%TZ)" >> $S
