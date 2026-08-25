#!/usr/bin/env bash
# RF3 fused-HiFi arm: the owed accuracy legs, on pc card 0.
#
# Every rung carries its OWN a1 on this card, run TWICE in two processes, because pc card 0
# silently miscomputes some ttnn matmuls at a low location-keyed rate
# (`pc-card0-512aa-fold-nondeterminism`) and a qb2 baseline is a different denominator
# (the committed rows in the state doc are qb2's). a1_p1 vs a1_p2 IS the noise floor, measured
# here, and no a5-minus-a1 difference smaller than it is a result.
#
# Not benchlocked: accuracy legs are not timed. They must not run while a timed leg does.
set -u
WT=/home/moritz/.coworker/wt/rf3-fused-hifi-precision-arm
PY=/home/moritz/tt-bio/env/bin/python3
PP=$WT:/home/moritz/rf3_perf_deps
R=$WT/perf/rf3/results
L=$R/logs
cd "$WT"
LEASE="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rf3-fused-hifi-precision-arm"

leg() { # leg <fixture> <refcache> <arm> <tag>
  local fx=$1 rc=$2 arm=$3 tag=$4
  [ -s "$R/pc_${tag}.json" ] && { echo "skip $tag (have it)"; return 0; }
  echo "=== $(date -Is) START $tag"
  env PYTHONPATH=$PP $LEASE timeout 2400 $PY -u scripts/rf3_port/accuracy_cell.py \
      --fixture "$fx" --arm "$arm" --seeds 0,1,2,3,4 \
      --ref-cache "$R/$rc" --work "$R/pc_${tag}" --out "$R/pc_${tag}.json" \
      > "$L/pc_${tag}.log" 2>&1
  echo "=== $(date -Is) END $tag rc=$?"
}

# The aligned-length control. The ragged pad fires on nothing at 128, so the rung isolates the arm.
leg cdk2_128  accuracy_cdk2_128  a1  cdk128_a1_p1
leg cdk2_128  accuracy_cdk2_128  a1  cdk128_a1_p2
leg cdk2_128  accuracy_cdk2_128  a5  cdk128_a5

# The third rung. a9 comes along because it beat a5 on both anchors on qb2 and is a config the
# `tri_att_sdpa_hifi` boolean cannot select.
leg cdk2_298  accuracy_cdk2_298  a1  cdk298_a1_p1
leg cdk2_298  accuracy_cdk2_298  a1  cdk298_a1_p2
leg cdk2_298  accuracy_cdk2_298  a5  cdk298_a5
leg cdk2_298  accuracy_cdk2_298  a9  cdk298_a9

# The two anchors again, on a second host. Not pooled with qb2's rows -- read for the SIGN of
# a5-minus-a1, which is the claim qb2 made and the only part that should travel between cards.
leg 7roa_117  accuracy_7roa_117  a1  roa117_a1_p1
leg 7roa_117  accuracy_7roa_117  a1  roa117_a1_p2
leg 7roa_117  accuracy_7roa_117  a5  roa117_a5
leg 7roa_117  accuracy_7roa_117  a7  roa117_a7
leg 7roa_117  accuracy_7roa_117  a9  roa117_a9
leg 7roa_117  accuracy_7roa_117  a10 roa117_a10
leg ubq_76    accuracy_ubq_76    a1  ubq76_a1_p1
leg ubq_76    accuracy_ubq_76    a1  ubq76_a1_p2
leg ubq_76    accuracy_ubq_76    a5  ubq76_a5
leg ubq_76    accuracy_ubq_76    a7  ubq76_a7
echo "ALL ACC LEGS DONE $(date -Is)"

# Third process per rung: rule 2 of the 2026-08-25 amendment wants N>=3 for the on-card floor,
# and 7ROA's a1_p1/a1_p2 pair already disagreed on seed 2, so two is not enough here. The arm
# repeats are for the same reason: if a1 can move, so can a5, and a single a5 reading would carry
# the card's fault as if it were the arm's.
leg 7roa_117  accuracy_7roa_117  a1  roa117_a1_p3
leg 7roa_117  accuracy_7roa_117  a5  roa117_a5_p2
leg ubq_76    accuracy_ubq_76    a1  ubq76_a1_p3
leg ubq_76    accuracy_ubq_76    a5  ubq76_a5_p2
leg cdk2_298  accuracy_cdk2_298  a1  cdk298_a1_p3
leg cdk2_298  accuracy_cdk2_298  a5  cdk298_a5_p2
leg cdk2_128  accuracy_cdk2_128  a1  cdk128_a1_p3
echo "P3 LEGS DONE $(date -Is)"
