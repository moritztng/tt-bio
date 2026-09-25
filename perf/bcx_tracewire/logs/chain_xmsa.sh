#!/bin/bash
# bcx-tracewire, qb2 card 3: the three-arm interleave with the extra-MSA swap on, then the
# two-process digest pair on the merged code (26fc95580 changed the backward's slice path).
cd /home/ttuser/.coworker/wt/bcx-tracewire || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
L=perf/bcx_tracewire/logs
S=$L/chain_xmsa.status
: > $S
timeout 3000 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --interleave --extra-msa --rounds 26 \
  --seed 100 --card 3 --project perf/bcx_tracewire/runs/interleave3_xmsa_fenced_seed100 \
  --out round_interleave3_xmsa_fenced_seed100.json > $L/interleave3_xmsa_fenced_s100.log 2>&1 < /dev/null
echo "interleave exit $? $(date -u +%FT%TZ)" >> $S
for arm in trace eager; do
  timeout 1800 $PY -X faulthandler perf/bcx_tracewire/round_ab.py --arm $arm --digest --extra-msa \
    --rounds 7 --seed 100 --card 3 --project perf/bcx_tracewire/runs/digest_xmsa_fenced_${arm}_seed100 \
    --out round_digest_xmsa_fenced_${arm}_seed100.json > $L/digest_xmsa_fenced_${arm}_s100.log 2>&1 < /dev/null
  echo "$arm exit $? $(date -u +%FT%TZ)" >> $S
done
echo "chain done $(date -u +%FT%TZ)" >> $S
