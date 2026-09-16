#!/usr/bin/env bash
# Flag-off control for ONE capacity cell, for attributing a 1536 aa FAIL.
#
# The gate runs each cell once, on whatever the tree defaults to. That is enough for a PASS -- a
# pass is a pass -- but a FAIL on the default-ON tree says nothing on its own, because this tree
# carries everyone else's merged levers and the cell may fail on main too. Same tree, same card,
# same cell, one env var different, one arm per PROCESS.
#
# Usage: cap_offarm.sh <model> [card]
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2
cd "$WT" || exit 1
M="${1:?model}"; CARD="${2:-3}"
OUT=perf/ttx_a3/gate6
PROG=$OUT/progress
P=/home/ttuser/tt-bio-dev/env/bin/python3
export PYTHONPATH="$WT"
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS="$CARD"
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge2
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

name="capoff-$M"
grep -q " $name rc=" "$PROG" && { echo "already recorded: $name"; exit 0; }
log "$name START card=$CARD loadavg=$(cut -d' ' -f1 /proc/loadavg)"
env TT_BIO_SDPA_FUSED_LARGE_S=0 TT_VISIBLE_DEVICES=$CARD timeout -k 30 43200 \
  $P scripts/capacity_gate.py --models "$M" --workers "tt-quietbox2:$CARD" --no-card-reset \
    --work-dir "$OUT/capoff-$M" --report "$OUT/capacity_off_$M.json" \
    > "$OUT/$name.log" 2>&1
log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
