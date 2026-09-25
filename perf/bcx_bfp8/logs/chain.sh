#!/bin/bash
# bcx-bfp8: traced design round, bf16 against bfp8 activations, one process per arm, alternating
# so load drift lands on both. qb2 card 3, extra-MSA swap on, seed 100 (bcx-tracewire's unit).
cd /home/ttuser/.coworker/wt/bcx-bfp8 || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_bfp8/logs; S=$L/chain.status; : > $S
for leg in bf16_a b8_a bf16_b b8_b; do
  flag=""; [[ $leg == b8* ]] && flag="--b8"
  timeout 1100 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm trace $flag --extra-msa \
    --rounds 12 --seed 100 --card 3 --project perf/bcx_bfp8/runs/round_$leg \
    --out ../bcx_bfp8/round_trace_${leg}_seed100.json > $L/round_$leg.log 2>&1 < /dev/null
  echo "$leg exit $? $(date -u +%FT%TZ)" >> $S
done
echo "chain done $(date -u +%FT%TZ)" >> $S
