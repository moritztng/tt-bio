#!/bin/bash
# Run C: the cross-model regression arm for the `_qkv_mm_config` guard relaxation. protenix-v2 must
# be byte-identical between `on` and `k1def` -- its widths are kt in {2,8} and never reach the new
# (12,36)/(12,12) entries, so the identity-scoped skip must be invisible here.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model protenix-v2 --sizes 512 \
      --arms on,k1def,on --out perf/odde4x/ab_px_k1def.json
echo "RC=$?"
