#!/bin/bash
# Phase 8 inner -- everything this row still owes on the p300c, inside ONE lock hold, with a
# control whose answer on this card is already known placed on BOTH sides of it.
#
# Two defects in phase 6 that this replaces:
#
#  1. It folded the old arm FIRST and the control LAST, so a bad window cost two full cells inside
#     the lock before anything revealed it was bad. The control is cheap (2 folds of a 14.4 s tree)
#     and it is the direct measurement of the thing loadavg only proxies, so it goes first and the
#     lock is released in ~90 s when the window is no good instead of ~10 min.
#  2. A control only on one side dates the window at one end. Opening AND closing controls bracket
#     the expensive cells, so a foreign job that starts mid-pair is caught rather than banked.
#
# The same bracket also buys Protenix its first honest p300c cell. Phase 7 had to fall back on the
# loadavg proxy because Protenix has never been measured on this part and so has no known answer to
# check against. It does not need its own: boltz2 on `main` reads 14.360 s on this card
# (c14-land-tail is independently at 14.308 s over 36 folds), and a window good enough for that
# fold is good enough for the Protenix fold taken between its two readings.
set -u
TAG=${TAG:?}
OLD=/home/ttuser/pvx_qb2/old
NEW=/home/ttuser/pvx_qb2/new
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
OUT=${OUT:-/home/ttuser/pvx_qb2/out3}
REF=${REF:-14.360}
TOL=${TOL:-0.06}
ENTER_MAX=${ENTER_MAX:-10.0}
SETTLE=${SETTLE:-60}
LOG=${LOG:-/home/ttuser/pvx_qb2/p8_cells.log}

med(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[\"summary\"][\"median_fold_s\"])" "$1" 2>/dev/null; }

# cell <tree> <model> <reps> <tag>
cell(){
  echo "    $(date -u +%H:%M:%SZ) $4  ($2, $3 timed folds after a cold one)"
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=2,3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
      PYTHONPATH="$1" \
    "$PY" -u "$1/perf/pvx_baseline/cell.py" --model "$2" --reps "$3" --clock 1350 \
      --out "$OUT/$4.json" --tag "$4" >>"$LOG" 2>&1
}

# control_ok <file> <which>
control_ok(){
  local m; m=$(med "$1")
  [ -n "$m" ] || { echo "    control $2: no summary, treating as a failed window"; return 1; }
  awk -v n="$m" -v r="$REF" -v t="$TOL" -v w="$2" \
    "BEGIN{d=(n-r)/r; if(d<0)d=-d; printf \"    control %s: %.3f s against %.3f s, %+.2f %%\n\", w, n, r, 100*(n-r)/r; exit !(d<=t)}"
}

echo "    took the lock at $(date -u +%H:%M:%SZ), loadavg $(cut -d" " -f1 /proc/loadavg), settling ${SETTLE}s"
# /proc/loadavg field 1 is a 1-MINUTE average, so read at hand-off it describes the PREVIOUS
# holder. Settle before reading it; 60 s against a ~10 min hold is not squatting.
sleep "$SETTLE"
l=$(cut -d" " -f1 /proc/loadavg)
echo "    settled at $(date -u +%H:%M:%SZ), loadavg $l, bar $ENTER_MAX"
awk -v a="$l" -v b="$ENTER_MAX" "BEGIN{exit !(a+0<=b+0)}" || {
  echo "    loadavg $l over $ENTER_MAX, releasing the lock instead of squatting on it"; exit 75; }

cell "$NEW" boltz2 2 "ctl_open_$TAG" || { echo "    opening control did not run"; exit 1; }
control_ok "$OUT/ctl_open_$TAG.json" open || {
  echo "    window refused BEFORE the expensive cells, releasing the lock"; exit 76; }

cell "$OLD" boltz2      4 "b2_old_$TAG"     || { echo "    b2_old did not run";     exit 1; }
cell "$NEW" protenix-v2 4 "ptx_p300c_$TAG"  || { echo "    ptx_p300c did not run";  exit 1; }
cell "$NEW" boltz2      4 "b2_pairnew_$TAG" || { echo "    b2_pairnew did not run"; exit 1; }

control_ok "$OUT/b2_pairnew_$TAG.json" close || {
  echo "    the window degraded during the pair, parking it"; exit 77; }
echo "    $(date -u +%H:%M:%SZ) $TAG folded, both controls held"
