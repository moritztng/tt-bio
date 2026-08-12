#!/bin/sh
# EXEC chain for esmfold2-to-4x: the config sweep and THE fold A/B under ONE benchlock hold, so
# no other worker can slip a fold in between and shift the baseline. Both write incrementally.
set -x
cd /home/ttuser/.coworker/wt/esmfold2-to-4x || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:esmfold2-to-4x
$PY perf/esm4x/mmcfg_sweep.py --out perf/esm4x/mmcfg_sweep_512_c2.json
$PY perf/esm4x/fold_ab.py --arms base,armc,both --rounds 2 \
    --inproj-from perf/esm4x/mmcfg_sweep_512_c2.json \
    --out perf/esm4x/fold_ab_512_c2.json
echo "CHAIN DONE rc=$?"
