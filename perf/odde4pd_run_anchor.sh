#!/bin/bash
# The uninstrumented anchor on origin/main, qb2 card 2, under benchlock. The perf page cell (87.1 s)
# is DERIVED (90.874 - 3.775 in-instrument); this measures the wall main actually delivers today.
set -u
WT=/home/ttuser/.coworker/wt/opendde-to-4x-per-dollar
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:opendde-to-4x-per-dollar PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm BENCHLOCK_MAXLOAD=0.6
PY=/home/ttuser/tt-bio-dev/env/bin/python3
FIX=$WT/perf/size512/fixtures
O=$WT/perf/odde4pd
mkdir -p $O
echo "=== anchor: origin/main, uninstrumented, 3 warm $(date -Is) ==="
/home/ttuser/.coworker/scripts/benchlock.sh opendde-to-4x-per-dollar -- \
  $PY -u scripts/gpu_vs_tt/tt_baseline.py --model opendde --repeat 3 \
      --target $FIX/cdk2x2_512.yaml --msa-a3m $FIX/cdk2x2_512.a3m \
      --label "512 aa cdk2x2, origin/main anchor" \
      --msa-dir $WT/.msa_odde4pd --out $O/anchor_main_512_c2.json
echo "RC=$? $(date -Is)"
