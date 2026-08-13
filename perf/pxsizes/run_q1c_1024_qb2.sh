#!/bin/bash
# qb2 1024 aa: the one row still owed from the qb2 size curve (state 14.6).
#
# Pass 2 left this row unmeasured. The `on` p1 arm was launched 17:23:07Z on the 15:48Z boot and was
# killed by its own 1300 s timeout at 17:44:47Z (RC 124) without producing a JSON; the `base4` p1 arm
# that followed inherited a card the SIGTERM had left wedged (12.6: killing a stalled fold wedges the
# card, enumeration is not a liveness check) and ran until the host was power-cycled. So pass 2
# produced no 1024 datapoint and no attributable stall onset: the second arm's failure is explained by
# the first arm's kill, not by anything about 1024.
#
# What that leaves unresolved, and what this leg tests: whether a 1024 aa fold on qb2 card 2 completes
# at all as the FIRST fold in its process. Expectation from the measured 768 row: qb2 768 = 134.7 s on
# the `on` arm against qb1's 155.4 s, and qb1 1024 = 288.0 s, so qb2 1024 should land near 250-300 s.
# 1300 s is 4.5x that, which is why pass 2's overrun reads as a stall and not as a slow fold.
#
# Instrument unchanged from the 768 leg (state 14.1): one arm per process, --skip-cold, order
# on, base4, on, base4, ratio from the SECOND process of each arm, stop rule = if either arm's
# cross-process spread exceeds one third of the (base4 - on) gap the ratio is NOT RESOLVED.
#
# Per-arm timeout 700 s, down from pass 2's 1300 s. 700 s is >2x the 250-300 s expectation, so a
# healthy fold has ample room, and a stall costs 12 min instead of 22. On any nonzero RC the card is
# reset with tt-smi -r 2 before the next arm starts, which is the step pass 2 did not take.
set -u
WT=/home/ttuser/.coworker/wt/protenix-v2-qb2-768-1024-rows
cd $WT
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=worker:protenix-v2-qb2-768-1024-rows PYTHONPATH=$WT
export ESM_ROOT=/home/ttuser/esm
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
SMI=/home/ttuser/.tenstorrent-venv/bin/tt-smi
O=perf/pxsizes
SELF=$WT/perf/pxsizes/run_q1c_1024_qb2.sh
TMO=700

one () {  # one <arm> <tag>
  echo "=== $(date -Is)  1024 aa arm $1 $2 (one arm, one process) ==="
  timeout $TMO $PY -u perf/size512/fold_ab512.py --sizes 1024 --arms "$1" --skip-cold \
      --timers full --out "$O/q1c_1024_$1_$2.json"
  rc=$?
  echo "RC_1024_$1_$2=$rc  at $(date -Is)"
  if [ $rc -ne 0 ]; then
    echo "--- nonzero RC, resetting card 2 before the next arm ($(date -Is)) ---"
    $SMI -r 2 2>&1 | tail -5
    echo "--- reset returned $? at $(date -Is) ---"
  fi
}

if [ "${1:-}" = "--leg" ]; then
  one on    p1
  one base4 p1
  one on    p2
  one base4 p2
  exit 0
fi

echo "=== host: $(hostname)  card: $TT_VISIBLE_DEVICES  main: $(git log --oneline -1)  start: $(date -Is) ==="
echo "=== uptime: $(uptime) ==="
sha256sum $WT/perf/size512/fixtures/cdk2x2_1024.yaml $WT/perf/size512/fixtures/cdk2x2_1024.a3m
$PY -c "import importlib.metadata as im; print('ttnn', im.version('ttnn'))"
$BL protenix-v2-qb2-768-1024-rows -- bash $SELF --leg
echo "=== 1024 leg done: $(date -Is) ==="
