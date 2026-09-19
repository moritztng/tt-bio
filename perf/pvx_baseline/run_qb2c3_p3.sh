#!/bin/bash
# Phase 3 — a retry ring behind phase 2.
#
# Phase 2 gives each cell ONE benchlock attempt. Its first attempt (b2_old_s1, pid 17970) was
# launched at 19:42Z with BENCHLOCK_WAIT_S=25200, so it expires at 02:42Z, and c14-land-tail has
# held the lock since 19:40:33Z with a 1h45m stall already inside it (block 4 finished 21:47Z,
# block 5 did not start until 23:32Z). If that repeats, b2_old_s1 times out with rc=75, phase 2
# moves on, and the single most important remaining cell is never retried.
#
# run() skips any tag that already carries a summary, so this ring only picks up what phase 2 lost.
# It also keeps exactly one harvester alive, so a cell that lands after a worker turn ends is still
# committed and pushed.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 32400 ))}

echo "=== $(date -u +%FT%TZ) phase 3 armed, deadline $(date -u -d @$DEADLINE +%FT%TZ) ==="
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
  echo "=== $(date -u +%H:%M:%SZ) phase-2 chain (pid $WAITPID) exited ==="
fi

keep_harvester() {
  pgrep -f "bash /home/ttuser/pvx_qb2/harvest.sh" >/dev/null 2>&1 && return 0
  local left=$(( DEADLINE - $(date +%s) + 1800 ))
  [ "$left" -gt 300 ] || return 0
  echo "=== $(date -u +%H:%M:%SZ) no harvester alive, starting one for ${left}s ==="
  ( cd /home/ttuser/.coworker/wt/pvx-baseline && \
    HARVEST_S=$left setsid nohup bash /home/ttuser/pvx_qb2/harvest.sh >>/home/ttuser/pvx_qb2/harvest_p7.log 2>&1 & )
}

run() {
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  if [ -s "$OUT/$tag.json" ] && grep -q "\"summary\"" "$OUT/$tag.json" 2>/dev/null; then
    echo "=== $(date -u +%H:%M:%SZ) $tag already summarised, skipping ==="; return 0
  fi
  local left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 900 ]; then echo "=== $(date -u +%H:%M:%SZ) past deadline, stopping before $tag ==="; return 1; fi
  keep_harvester
  echo "=== $(date -u +%H:%M:%SZ) RETRY $tag tree=$tree model=$model reps=$reps clock=$clk budget=${left}s ==="
  BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=7200 "$BL" pvx-baseline -- \
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock "$clk" \
      --out "$OUT/$tag.json" --tag "$tag" "$@"
  echo "RC=$? $tag"
  keep_harvester
}

# Same order as phase 2: the p300c ratio first, then Protenix on the anchors own board class,
# then the second sessions that turn each into a cross-session A/A floor.
run "$OLD" b2_old_s1  boltz2      4 1350
run "$NEW" ptx_new_s1 protenix-v2 4 1350
run "$NEW" b2_new_s2  boltz2      4 1350
run "$OLD" b2_old_s2  boltz2      4 1350
run "$NEW" ptx_new_s2 protenix-v2 4 1350
keep_harvester
echo "PVXBASELINEP3DONE $(date -u +%FT%TZ)"
