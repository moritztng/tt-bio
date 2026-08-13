#!/bin/bash
# qb2 768 + 1024 aa: the two rows the size curve never got on its native host.
#
# state/protenix-v2-sizes-perf.md has the qb2 curve complete at 128/256/512/640 and both large rows
# missing (2.2). They were taken on qb1 instead -- 13.3: 768 = 1.0540x, 1024 = 1.0524x -- and cannot
# be moved across: qb1 is p150a / ttnn 0.67.4 / 13x10, qb2 is ttnn 0.68.0 / 11x10, and E6's gate is
# grid-sensitive (the reblock_permute lesson). 11.6 names the gap: no qb1 point at a size qb2 also
# measured, so the two curves cannot be calibrated against each other.
#
# Instrument: one arm per process, --skip-cold, exactly 13.1. Section 12 makes that permanent, not a
# qb2 workaround -- the second-fold MSA stall crosses board and ttnn version and only its onset moves.
#
# TWO processes per arm, not one. 12.2 shows no `base4` arm has ever COMPLETED at 768 or 1024 on qb2
# (both multi-arm legs died in fold 2 before reaching it), so base4's fallback kernels -- the ttnn
# path K1/K1b displace -- have never JIT-compiled at these shapes on this box. 11.3 measured
# first-ever compile at 71.70 s (768) and 80.99 s (1024) against a cache-warm per-process cold of
# 7.55 s / 5.52 s. An uncontrolled first-compile landing in base4 would manufacture a win of exactly
# the size being claimed (the qb1 gaps are 8.4 s and 15.1 s). So each arm runs twice and the ratio is
# taken from the SECOND process of each arm, with |p1 - p2| per arm reported as the compile magnitude
# and as that arm's own cross-process floor. Order is on, base4, on, base4 so the two are bracketed.
#
# One benchlock hold per size (11's convention), so the 1024 leg re-queues rather than holding the
# box for both.
#
# REGISTERED PREDICTIONS (committed before the run; not edited afterwards):
#   Q1  Per-process fold: 768 lands 150-215 s, 1024 lands 270-350 s. qb2's own first-ever folds at
#       these sizes were 199.63 s and 315.09 s (7.2, 0.3) and both carried first-compile; qb1's
#       cache-warm per-process numbers are 155.4 s and 288.0 s on a different grid.
#   Q2  |p1 - p2| for the `on` arm is under 10 s at both sizes (qb1's cross-process A/A floors were
#       0.550 s and 0.223 s). For `base4` it may be much larger at 768/1024 for the compile reason
#       above; if it is, p1 is the discard and p2 is the number.
#   Q3  base4/on at 768 on qb2 is 1.04-1.12x, and at 1024 1.04-1.12x -- same direction as qb1's
#       1.0540x / 1.0524x, with room above because on qb1 E6 and K2 both serve 0 at these sizes
#       (11.4), so that ratio is K1+K1b alone. If either gate opens on the 11x10 grid, qb2's `on` arm
#       carries a lever qb1's did not.
#   Q4  E6 (`gated_kernel`) is enabled and serves 0 at BOTH sizes on qb2 as well. Then "gated off in
#       the shipped stack above 640 aa" is a property of the code, not of qb1's grid.
#   Q5  K2 (`persistent_mask`) is enabled and serves 0 at both sizes on qb2 as well, declining on
#       fill_preconditions as it does on qb1.
#   Q6  >= 95 % of the (base4 - on) fold gap is body:TriangleAttention at both sizes, as at 128, 256
#       (2.2), 768 (11.1: 99.3 %) and 1024 (13.2: 97.9 %).
#   Q7  Parity: one CIF sha256 per size across all four arms, and plDDT equal to the value both hosts
#       already recorded -- 0.787723 at 768, 0.778516 at 1024. Anything else means main has moved
#       numerically since those folds, which would be a finding of its own.
#   Q8  Every process returns: one arm per process means no fold here is ever a second fold.
#   Stop rule: if either arm's cross-process spread at a size exceeds one third of that size's
#   (base4 - on) gap, the ratio is reported NOT RESOLVED rather than as a number.
#   Stop rule: a process that runs past its timeout is killed by EXPLICIT pid and the card is then
#   reset with tt-smi -r 2 -- 12.6 measured that killing a stalled fold leaves the card wedged while
#   tt-smi -ls still enumerates it, so enumeration is not a liveness check.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-qb2-768-1024-rows
cd $WT
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-v2-qb2-768-1024-rows PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/pxsizes
SELF=$WT/perf/pxsizes/run_q1b_qb2.sh

one () {  # one <size> <arm> <tag> <timeout>
  echo "=== $(date -Is)  $1 aa arm $2 $3 (one arm, one process) ==="
  timeout $4 $PY -u perf/size512/fold_ab512.py --sizes $1 --arms "$2" --skip-cold \
      --timers full --out "$O/q1b_$1_$2_$3.json"
  echo "RC_$1_$2_$3=$?  at $(date -Is)"
}

if [ "${1:-}" = "--leg" ]; then
  one "$2" on    p1 "$3"
  one "$2" base4 p1 "$3"
  one "$2" on    p2 "$3"
  one "$2" base4 p2 "$3"
  exit 0
fi

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  main: $(git log --oneline -1)  start: $(date -Is) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_768.yaml $WT/perf/size512/fixtures/cdk2x2_768.a3m \
          $WT/perf/size512/fixtures/cdk2x2_1024.yaml $WT/perf/size512/fixtures/cdk2x2_1024.a3m
$PY -c "import importlib.metadata as im; print('ttnn', im.version('ttnn'))"

$BL protenix-v2-qb2-768-1024-rows -- bash $SELF --leg 768  900
echo "=== 768 leg done: $(date -Is) ==="
$BL protenix-v2-qb2-768-1024-rows -- bash $SELF --leg 1024 1300
echo "=== all done: $(date -Is) ==="
