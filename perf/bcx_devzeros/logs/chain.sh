#!/bin/bash
# bcx-devzeros, qb2 card 1. DEVICE_ZEROS graded on its own against the shipped device open
# (--no-wire), then its bit-identity on the loop path, then tracewire's three-arm configuration
# reproduced. Extra-MSA swap ON throughout, seed 100, pdl1.
cd /home/ttuser/.coworker/wt/bcx-devzeros || exit 1
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-devzeros
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
PY=/home/ttuser/bcx_e2e_venv/bin/python3
R=perf/bcx_tracewire/round_ab.py
L=perf/bcx_devzeros/logs; S=$L/chain.status; : > $S
run() { name=$1; shift; timeout 2400 $PY -X faulthandler $R "$@" --extra-msa --seed 100 --card 1 \
  --project perf/bcx_devzeros/runs/$name > $L/$name.log 2>&1 < /dev/null
  echo "$name exit $? $(date -u +%FT%TZ)" >> $S; }
run nowire_interleave --no-wire --interleave --rounds 26 --out devzeros_nowire_interleave_seed100.json
run nowire_digest_zeros --no-wire --arm zeros --digest --rounds 7 --out devzeros_nowire_digest_zeros_seed100.json
run nowire_digest_eager --no-wire --arm eager --digest --rounds 7 --out devzeros_nowire_digest_eager_seed100.json
run wire_interleave3 --interleave --rounds 26 --out devzeros_wire_interleave3_seed100.json
echo "chain done $(date -u +%FT%TZ)" >> $S
