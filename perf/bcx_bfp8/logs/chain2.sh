#!/bin/bash
# bcx-bfp8 pass 2, qb2 card 3. Why the bfp8 trajectory ends after two rounds, then the
# correctness floor: every graded block's VJP against a float64 reference, both arms.
cd /home/ttuser/.coworker/wt/bcx-bfp8 || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-bfp8
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_bfp8/logs; S=$L/chain2.status; : > $S
for leg in b8 bf16; do
  flag=""; [[ $leg == b8 ]] && flag="--b8"
  timeout 900 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm trace $flag --finite \
    --extra-msa --rounds 12 --seed 100 --card 3 --project perf/bcx_bfp8/runs/finite_$leg \
    --out ../bcx_bfp8/round_finite_${leg}_seed100.json > $L/finite_$leg.log 2>&1 < /dev/null
  echo "finite_$leg exit $? $(date -u +%FT%TZ)" >> $S
done
for B in 1 0; do
  BFP8_B8=$B timeout 1700 $PY perf/bcx_bfp8/grade.py --card 3 --n 128 --blocks 0,1,26 \
    --seed 0 --out grade_b8$B.json > $L/grade_b8$B.log 2>&1 < /dev/null
  echo "grade_b8$B exit $? $(date -u +%FT%TZ)" >> $S
done
echo "chain2 done $(date -u +%FT%TZ)" >> $S
