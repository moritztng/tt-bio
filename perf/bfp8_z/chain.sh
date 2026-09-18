#!/bin/bash
# Accuracy legs for bfp8-z-accumulator: one process per dtype arm, 512 aa, seeds 0-3, each leg
# closing with its own A/A repeat. One process per arm is required, not a style choice --
# sdpa_generic's program cache key omits every operand dtype but q's on this tree.
set -u
cd /home/ttuser/.coworker/wt/bfp8-z-accumulator || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_CARDS=0,3
export TT_BIO_LEASE_HOLDER=worker:bfp8-z-accumulator
export TT_METAL_HOME=${TT_METAL_HOME:-}
O=perf/bfp8_z
for leg in base Z; do
  echo "=== leg $leg $(date -u +%H:%M:%SZ)"
  timeout 2400 $PY $O/fold_z.py --out $O/acc_${leg}.json --cifdir $O/cif_${leg} \
      --sizes 512 --plan ${leg}:0,1,2,3 2>&1
  echo "=== leg $leg rc=$? $(date -u +%H:%M:%SZ)"
done
echo "=== chain done $(date -u +%H:%M:%SZ)"
