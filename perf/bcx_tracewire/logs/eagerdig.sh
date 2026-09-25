#!/bin/bash
cd /home/ttuser/.coworker/wt/bcx-tracewire || exit 1
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-tracewire
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
L=perf/bcx_tracewire/logs
timeout 1200 /home/ttuser/bcx_e2e_venv/bin/python3 -X faulthandler perf/bcx_tracewire/round_ab.py --arm eager --digest --extra-msa \
  --rounds 7 --seed 100 --card 3 --project perf/bcx_tracewire/runs/final_digest_eager_seed100 \
  --out round_digest_xmsa_final_eager_seed100.json > $L/final_digest_eager.log 2>&1 < /dev/null
echo "eager exit $? $(date -u +%FT%TZ)" > $L/eagerdig.status
