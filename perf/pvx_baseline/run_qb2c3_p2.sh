#!/bin/bash
# Phase 2 of the p300c pair, card 3, pinned 1350 MHz.
#
# Reordered against run_qb2c3.sh: the box is contended by three other PVX rows and a release
# gate, so one session of each of the three cells lands before any second session. With
# b2_new_s1 (14.360 s) already on disk, b2_old_s1 completes the p300c ratio and ptx_new_s1
# puts Protenix on the board class its 54.760 s anchor was actually taken on. The second
# sessions are what turn each into a cross-session A/A floor, and they come after.
#
# Waits on the phase-1 benchlock waiter so two pvx-baseline waiters cannot queue against
# each other, per the pass-2 pattern.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
mkdir -p "$OUT"

WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  echo "=== $(date -u +%H:%M:%SZ) waiting on phase-1 waiter pid $WAITPID ==="
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
  echo "=== $(date -u +%H:%M:%SZ) phase-1 waiter gone ==="
fi

run() {
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q "\"summary\"" "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already has a summary, skipping ==="
    return 0
  fi
  echo "=== $(date -u +%H:%M:%SZ) $tag  tree=$tree model=$model reps=$reps clock=$clk $* ==="
  BENCHLOCK_WAIT_S=25200 BENCHLOCK_LOAD_WAIT_S=7200 "$BL" pvx-baseline -- \
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock "$clk" \
      --out "$OUT/$tag.json" --tag "$tag" "$@"
  echo "RC=$? $tag"
}

run "$OLD" b2_old_s1  boltz2     4 1350
run "$NEW" ptx_new_s1 protenix-v2 4 1350
run "$NEW" b2_new_s2  boltz2     4 1350
run "$OLD" b2_old_s2  boltz2     4 1350
run "$NEW" ptx_new_s2 protenix-v2 4 1350

echo "PVXBASELINEDONE $(date -u +%FT%TZ)"
