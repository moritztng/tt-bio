#!/bin/bash
# Part B -- the ring-2 re-adjudication, exactly as pre-registered in
# ~/.coworker/state/wh-boltz2-640aa-clash-rootcause_PREREG.md. Nothing here may be edited once
# the first CIF is scored.
#
# 12 targets x 5 arms = 60 folds, one Wormhole card, strictly sequential. Order is chosen so that
# a chain cut short still adjudicates something: all 12 targets get `base` first, then `ship`
# (the corner actually on trial), then the two single-lever corners, then the null arm last.
# `null` is the yardstick, not a corner, and it is the most expensive leg (5 diffusion samples),
# so it goes at the end rather than starving the comparison it exists to calibrate.
#
#   base  K3 off  slmc 608   ds 1      (= whpre)
#   ship  K3 on   slmc 1088  ds 1      (= ring 2)
#   k3    K3 on   slmc 608   ds 1
#   cc    K3 off  slmc 1088  ds 1
#   null  K3 off  slmc 608   ds 5
#
# Band 768 is the gate's own A/A: K3 cannot act where 768 % 256 == 0, so `k3` must come back
# byte-identical to `base` there and `ship` byte-identical to `cc`. score.py records a CIF digest
# per fold for exactly that check.
set -u
B=/home/cust-team/mthuening/whb2clash
T=$B/tree
DEV=${WHB2_DEV:-28}
export WHB2_PY=/home/cust-team/mthuening/tt-bio/env/bin/python

B640="P22303 P27694 P03951 P20794 P17405 O14744"
B768="P54802 P42224 P18074 P47712 Q05823 O15111"
ALL="$B640 $B768"

mkdir -p $B/partb

leg () {  # arm k3 slmc samples
  local ARM=$1 K3=$2 SLMC=$3 NS=$4
  for ACC in $ALL; do
    local OUT=$B/partb/${ACC}_${ARM}
    # Idempotent across relaunches: a leg that already produced a structure is not re-folded.
    if compgen -G "$OUT/**/*.cif" > /dev/null 2>&1; then
      echo "=== $ACC $ARM already done, skipping ==="
      continue
    fi
    echo "=== $ACC $ARM start $(date -u +%FT%TZ) ==="
    timeout 7200 $T/perf/whb2clash/run_arm.sh $OUT \
      $T/perf/whb2clash/fixtures/$ACC.yaml $K3 $SLMC $DEV $B/msa $NS > $OUT.log 2>&1
    echo "=== $ACC $ARM rc=$? $(date -u +%FT%TZ) ==="
  done
}

leg base 0 608  1
leg ship 1 1088 1
leg k3   1 608  1
leg cc   0 1088 1
leg null 0 608  5
echo "PARTB DONE $(date -u +%FT%TZ)"
