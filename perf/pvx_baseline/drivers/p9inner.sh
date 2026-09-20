#!/bin/bash
# Phase 9 inner -- the p300c bracket, on the card this launch is granted (physical 1).
#
# What changed from phase 8, and why. Phase 8 folded on card 3 because the known-answer control
# 14.360 s is a property of card 3, and it never got a window: the ring launched at 07:07:41Z, sat
# in the benchlock queue behind pvx-orchestrator's size-ladder, and died in the 07:30Z reboot
# without folding once. This launch is granted card 1, so the control's provenance changes from
# same-card to CROSS-CARD and is recorded as such: 14.360 s (card 3, this row) and 14.308 s over
# 36 folds (c14-land-tail, card 2) are two readings of the same tree on two chips of the same
# p300c box, and a card-1 reading inside 6 % of them admits the window AND cross-checks the chip.
# If it lands outside, that is a card fact to report, not a window to refuse silently -- so the
# refusal prints the number.
#
# The bracket itself is unchanged and is the part that matters: a cheap opening control admits or
# refuses the window before an expensive fold is spent, and the closing control is the new arm
# itself, so a foreign job that starts mid-pair is caught rather than banked.
set -u
TAG=${TAG:?}
OLD=/home/ttuser/pvx_qb2/old
NEW=/home/ttuser/pvx_qb2/new
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
OUT=${OUT:-/home/ttuser/pvx_qb2/out9}
CARD=${CARD:-1}
REF=${REF:-14.360}
TOL=${TOL:-0.06}
BRACKET_TOL=${BRACKET_TOL:-0.02}
ENTER_MAX=${ENTER_MAX:-3.0}
SETTLE=${SETTLE:-30}
LOG=${LOG:-/home/ttuser/pvx_qb2/p9_cells.log}
mkdir -p "$OUT"

med(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['median_fold_s'])" "$1" 2>/dev/null; }

cell(){
  echo "    $(date -u +%H:%M:%SZ) $4  ($2, $3 timed folds after a cold one, card $CARD)"
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
      PYTHONPATH="$1" \
    "$PY" -u "$1/perf/pvx_baseline/cell.py" --model "$2" --reps "$3" --clock 1350 \
      --out "$OUT/$4.json" --tag "$4" >>"$LOG" 2>&1
}

control_ok(){
  local m; m=$(med "$1")
  [ -n "$m" ] || { echo "    control $2: no summary, treating as a failed window"; return 1; }
  awk -v n="$m" -v r="$REF" -v t="$TOL" -v w="$2" \
    'BEGIN{d=(n-r)/r; if(d<0)d=-d; printf "    control %s: %.3f s against the cross-card %.3f s, %+.2f %%\n", w, n, r, 100*(n-r)/r; exit !(d<=t)}'
}

echo "    took the lock at $(date -u +%H:%M:%SZ), loadavg $(cut -d' ' -f1 /proc/loadavg), settling ${SETTLE}s"
sleep "$SETTLE"
l=$(cut -d' ' -f1 /proc/loadavg)
echo "    settled at $(date -u +%H:%M:%SZ), loadavg $l, bar $ENTER_MAX"
awk -v a="$l" -v b="$ENTER_MAX" 'BEGIN{exit !(a+0<=b+0)}' || {
  echo "    loadavg $l over $ENTER_MAX, releasing the lock instead of squatting on it"; exit 75; }

cell "$NEW" boltz2 2 "ctl_open_$TAG" || { echo "    opening control did not run"; exit 1; }
control_ok "$OUT/ctl_open_$TAG.json" open || {
  echo "    window refused BEFORE the expensive cells, releasing the lock"; exit 76; }

cell "$OLD" boltz2      4 "b2_old_$TAG"     || { echo "    b2_old did not run";     exit 1; }
cell "$NEW" protenix-v2 4 "ptx_p300c_$TAG"  || { echo "    ptx_p300c did not run";  exit 1; }
cell "$NEW" boltz2      4 "b2_pairnew_$TAG" || { echo "    b2_pairnew did not run"; exit 1; }

control_ok "$OUT/b2_pairnew_$TAG.json" close || {
  echo "    the window degraded during the pair, parking it"; exit 77; }

# The two controls are the same tree on the same card minutes apart. They bound the window's drift
# directly, without reference to another chip's number, and that is the tighter of the two tests.
o=$(med "$OUT/ctl_open_$TAG.json"); c=$(med "$OUT/b2_pairnew_$TAG.json")
awk -v a="$o" -v b="$c" -v t="$BRACKET_TOL" \
  'BEGIN{d=(b-a)/a; if(d<0)d=-d; printf "    bracket drift: open %.3f s -> close %.3f s, %+.2f %%\n", a, b, 100*(b-a)/a; exit !(d<=t)}' || {
  echo "    the two controls disagree by more than $BRACKET_TOL, parking the window"; exit 78; }
echo "    $(date -u +%H:%M:%SZ) $TAG folded, both controls held and the bracket is flat"
