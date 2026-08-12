#!/bin/bash
# The cross-model leak check for BOTH new entries, in one process:
#   k1def -- the `_qkv_mm_config` guard relaxation, scoped by object identity on `_MM_DEFAULT`
#   mm12  -- ask 4649's engine-leak requirement; run C of `opendde-512aa-deep-perf` only ever
#            covered g12, never the `_MM_BLOCK` entries
# protenix-v2 presents kt in {2, 8} and never (12,36)/(12,12), so both arms must be byte-identical
# to `on` at the full 64-hex digest and inside protenix's own A/A floor.
WT=/home/ttuser/.coworker/wt/opendde-to-4x
cd $WT
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x -- \
  $PY -u perf/other512/fold_ab_multi.py --model protenix-v2 --sizes 512 \
      --arms on,k1def,mm12,on --out perf/odde4x/ab_px_leak.json
echo "RC=$?"
