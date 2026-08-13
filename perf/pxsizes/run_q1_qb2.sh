#!/bin/bash
# qb2 768 + 1024 aa: the two rows the size curve never got on its native host.
#
# state/protenix-v2-sizes-perf.md has the qb2 curve complete at 128/256/512/640 and both large rows
# missing: 768 was "still owed" and 1024 "not obtainable in-process" (2.2), because the second fold
# in a process stalls the device in the MSA stack (7.1, 12.2 -- a protenix defect that crosses board
# and ttnn version; qb2 trips at fold 2, qb1 at fold 3 of 1024). The rows were taken on qb1 instead
# (11.1: 768 = 1.0540x, 13.1: 1024 = 1.0524x) and cannot be transferred: qb1 is p150a / ttnn 0.67.4 /
# 13x10, qb2 is ttnn 0.68.0 / 11x10, and E6s gate is grid-sensitive (the reblock_permute lesson).
#
# One arm per process, --skip-cold, exactly the 13.1 instrument. `on` runs FIRST at each size on
# purpose: main has moved since the qb2 rows were attempted (the triatt q-split now ships ON at
# <=1024 padded tokens), so some kernels may compile fresh, and any first-compile cost lands on the
# arm we claim the win FOR -- which understates the win instead of inventing one. The second `on`
# process at each size is both the cross-process A/A floor and the check for that confound.
#
# REGISTERED PREDICTIONS (committed before the run; not edited afterwards):
#   Q1  Per-process fold: 768 lands 150-210 s, 1024 lands 260-340 s. qb2s own first-ever folds were
#       199.63 s and 315.09 s (7.2, 0.3) and both carried first-compile. A process far above its
#       band means fresh kernel compilation, and then only the two `on` processes calibrate it.
#   Q2  base4/on at 768 on qb2 >= qb1s 1.0540x, predicted 1.05-1.12x, and at 1024 >= 1.0524x,
#       predicted 1.04-1.11x. Same direction as qb1 with room above it, because on qb1 E6 and K2
#       both serve 0 at these sizes (11.4) so the ratio there is K1+K1b alone; if either gate opens
#       on the 11x10 grid, qb2s `on` arm carries a lever qb1s did not.
#   Q3  E6 (`gated_kernel`) is enabled and serves 0 at BOTH sizes on qb2 as well. Gated off in the
#       shipped stack above 640 aa is then a property of the code, not of qb1s grid.
#   Q4  K2 (`persistent_mask`) serves 0 at both sizes on qb2 as well, declining on
#       fill_preconditions as it does on qb1.
#   Q5  >= 95 % of the (base4 - on) fold gap is body:TriangleAttention at both sizes, as at 128, 256
#       (2.2), 768 (11.1: 99.3 %) and 1024 (13.2: 97.9 %).
#   Q6  Parity: one CIF sha256 and one plDDT per size across arms, and the plDDT equals the value
#       both hosts already recorded -- 0.787723 at 768, 0.778516 at 1024. Anything else means main
#       has moved numerically since those folds, which would be a finding of its own.
#   Q7  Every process returns: one arm per process is the workaround for the 12.2 stall, so no fold
#       here is ever a second fold.
#   Stop rule: if the cross-process A/A floor at a size exceeds one third of that sizes
#   (base4 - on) gap, the ratio is reported NOT RESOLVED rather than as a number.
#   Stop rule: a process that runs past its timeout is killed by EXPLICIT pid and then the card is
#   reset with tt-smi -r 2 -- 12.6 measured that killing a stalled fold leaves the card wedged and
#   that tt-smi -ls still enumerates it, so enumeration is not a liveness check.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-qb2-768-1024-rows
cd $WT
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-v2-qb2-768-1024-rows PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
O=perf/pxsizes

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  main: $(git log --oneline -1)  start: $(date -Is) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_768.yaml $WT/perf/size512/fixtures/cdk2x2_768.a3m \
          $WT/perf/size512/fixtures/cdk2x2_1024.yaml $WT/perf/size512/fixtures/cdk2x2_1024.a3m

one () {  # one <size> <arm> <out> <timeout>
  echo "=== $(date -Is)  $1 aa arm $2 (one arm, one process) ==="
  timeout $4 $PY -u perf/size512/fold_ab512.py --sizes $1 --arms "$2" --skip-cold \
      --timers full --out "$3"
  echo "RC_$1_$2=$?  at $(date -Is)"
}

$BL protenix-v2-qb2-768-1024-rows -- bash -c '
  set -u
  cd '"$WT"'
  '"$(declare -f one)"'
  PY='"$PY"'; O='"$O"'
  one 768  on    $O/q1_768_on_p1.json    900
  one 768  base4 $O/q1_768_base4.json    900
  one 1024 on    $O/q1_1024_on_p1.json  1200
  one 1024 base4 $O/q1_1024_base4.json  1200
  one 768  on    $O/q1_768_on_p2.json    900
  one 1024 on    $O/q1_1024_on_p2.json  1200
'
echo "=== done: $(date -Is) ==="
