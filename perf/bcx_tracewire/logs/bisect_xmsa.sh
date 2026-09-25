#!/bin/bash
# bcx-tracewire: which arm leaves the loop's trajectory once the extra-MSA swap is on.
cd /home/ttuser/.coworker/wt/bcx-tracewire || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_tracewire/logs; S=$L/bisect_xmsa.status; : > $S
for arm in eager zeros trace; do
  rm -rf perf/bcx_tracewire/runs/bisect_xmsa_${arm}_seed100
  timeout 1200 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm $arm --digest --extra-msa \
    --rounds 3 --seed 100 --card 3 --project perf/bcx_tracewire/runs/bisect_xmsa_${arm}_seed100 \
    --out bisect_xmsa_${arm}_seed100.json > $L/bisect_xmsa_${arm}.log 2>&1 < /dev/null
  echo "$arm exit $? $(date -u +%FT%TZ)" >> $S
done
echo "chain done $(date -u +%FT%TZ)" >> $S
