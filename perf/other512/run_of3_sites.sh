#!/bin/bash
# The site-partition fold A/B for of3-fp32-dst-sdpa. One process, warm, alternating, the two `on`
# arms bracketing the session so the A/A floor is measured across the whole run rather than back
# to back. The confidence head stays fp32 on every arm (asserted in set_arm).
WT=/home/ttuser/.coworker/wt/of3-fp32-dst-sdpa
cd $WT || exit 1
export TT_VISIBLE_DEVICES=${CARD:-0} TT_BIO_LEASE_HOLDER=worker:of3-fp32-dst-sdpa PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm OF3_CKPT=/home/ttuser/of3-weights/of3-p2-155k.pt
export BENCHLOCK_WAIT_S=${BENCHLOCK_WAIT_S:-5400}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
SIZE=${SIZE:-512}
/home/ttuser/.coworker/scripts/benchlock.sh of3-fp32-dst-sdpa -- \
  $PY -u perf/other512/fold_ab_multi.py --model openfold3 --sizes $SIZE \
      --arms on,nofp32_msatmpl,nofp32_trunk,nofp32,on \
      --out perf/other512/ab_of3_sites_$SIZE.json
echo "RC=$?"
