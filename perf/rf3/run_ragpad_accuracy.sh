#!/usr/bin/env bash
# Phase 2: the ragged pad across RF3's accuracy ladder, not one target.
#
# Leg order is chosen so the cheapest leg validates the rest. Leg 1 is a1 with the pad OFF at
# ubq_76: if it reproduces the committed X = 0.1306 exactly, main has not moved RF3's coordinates
# since 25d0e40e and every a0/a1 number in state/rf3-fast-arm-accuracy.md is citable against
# today's tree. If it does not, nothing downstream may be compared to a committed row.
#
# cdk2_128 is on this list as the ALIGNED CONTROL, not as an accuracy rung: 128 divides 32, so the
# pad must fire on nothing and the coordinates must come back byte-identical. E7 of the predecessor
# doc retired it for scoring (a truncated kinase lobe has no unique fold, so both arms wander and
# the rung reads 4.5 A for the SHIPPED arm), which is exactly what makes it a good bit-exactness
# control and a bad verdict.
#
# The reference half is arm-independent and already committed, so each leg pays a featurisation and
# a device rollout, not a 234 s CPU trunk.
set -u
WT=/home/ttuser/.coworker/wt/rf3-4x-with-accuracy-land
PY=/home/ttuser/tt-bio-dev/env/bin/python3
PP="$WT:/home/ttuser/rf3_perf_deps"
LEASE="TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:rf3-4x-with-accuracy-land"
R=$WT/perf/rf3/results
CEN=$WT/perf/rf3/page512/census
cd "$WT" || exit 1

# Wait for the perf chain: one device context per process, and a fold under benchlock must not
# share the card with a diffusion rollout.
while ! grep -q RAGPAD_ARMS_DONE "$WT/perf/rf3/page512/ragpad_chain.status" 2>/dev/null; do sleep 20; done

leg() {  # leg <tag> <fixture> <refdir> <pad>
  tag=$1; fix=$2; ref=$3; pad=$4
  echo "=== $tag start $(date -u +%H:%M:%S) ==="
  env PYTHONPATH="$PP" $LEASE TT_BIO_SDPA_RAGGED_PAD=$pad \
      TT_BIO_SDPA_RAGGED_CENSUS="$CEN/ac_$tag" \
    "$PY" scripts/rf3_port/accuracy_cell.py --fixture "$fix" --arm a1 --seeds 0,1,2,3,4 \
      --ref-cache "$R/$ref" --work "$R/ac_$tag" --out "$R/ac_$tag.json" \
      > "$R/ac_$tag.log" 2>&1
  rc=$?
  echo "=== $tag exit $rc $(date -u +%H:%M:%S) ==="
  echo "$tag rc=$rc" >> "$R/ragpad_acc.status"
}

leg ubq76_a1off    ubq_76    accuracy_ubq_76    0
leg ubq76_a1on     ubq_76    accuracy_ubq_76    1
leg cdk128_a1on    cdk2_128  accuracy_cdk2_128  1
leg roa117_a1on    7roa_117  accuracy_7roa_117  1
leg cdk298_a1on    cdk2_298  accuracy_cdk2_298  1
leg roa117_a1off   7roa_117  accuracy_7roa_117  0
leg cdk298_a1off   cdk2_298  accuracy_cdk2_298  0
echo RAGPAD_ACC_DONE >> "$R/ragpad_acc.status"
echo "=== ALL DONE $(date -u +%H:%M:%S) ==="
