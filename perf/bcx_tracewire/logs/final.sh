#!/bin/bash
# bcx-tracewire final, qb2 card 3, on 628833035: trace digest (7 steps) then the 26-round
# three-arm interleave with the extra-MSA swap on.
cd /home/ttuser/.coworker/wt/bcx-tracewire || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_tracewire/logs; S=$L/final.status; : > $S
timeout 1200 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm trace --digest --extra-msa \
  --rounds 7 --seed 100 --card 3 --project perf/bcx_tracewire/runs/final_digest_trace_seed100 \
  --out round_digest_xmsa_final_trace_seed100.json > $L/final_digest_trace.log 2>&1 < /dev/null
echo "digest exit $? $(date -u +%FT%TZ)" >> $S
timeout 2400 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --interleave --extra-msa --rounds 26 \
  --seed 100 --card 3 --project perf/bcx_tracewire/runs/final_interleave_seed100 \
  --out round_interleave3_xmsa_final_seed100.json > $L/final_interleave.log 2>&1 < /dev/null
echo "interleave exit $? $(date -u +%FT%TZ)" >> $S
echo "chain done $(date -u +%FT%TZ)" >> $S
