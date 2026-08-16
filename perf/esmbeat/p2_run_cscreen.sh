#!/usr/bin/env bash
# Step 7: screen lever C (fused blocked pair FFN) before anyone builds it.
set -u
WT=/home/ttuser/.coworker/wt/esmfold2-beat-dgx-h200-p2
cd "$WT" || exit 1
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:esmfold2-beat-dgx-h200-p2
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export BENCHLOCK_WAIT_S=3600
/home/ttuser/.coworker/scripts/benchlock.sh esmfold2-beat-dgx-h200-p2 -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/esmbeat/s_c_assembly.py \
  --size 512 --out perf/esmbeat/s_c_assembly_512_c0.json
echo "CSCREEN_RC=$?"
