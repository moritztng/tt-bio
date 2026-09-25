#!/bin/bash
# bcx-bfp8: the traced trunk step, bf16 against bfp8, legs alternating so load drift lands on both.
cd /home/ttuser/.coworker/wt/bcx-bfp8 || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_bfp8/logs; S=$L/chain3.status; : > $S
for leg in a b; do
  for B in 0 1; do
    name=$([[ $B == 1 ]] && echo b8 || echo bf16)_$leg
    BFP8_B8=$B BFP8_LEG=$leg timeout 900 $PY -X faulthandler perf/bcx_bfp8/trunk_ab.py \
      > $L/trunk_$name.log 2>&1 < /dev/null
    echo "trunk_$name exit $? $(date -u +%FT%TZ)" >> $S
  done
done
echo "chain3 done $(date -u +%FT%TZ)" >> $S
