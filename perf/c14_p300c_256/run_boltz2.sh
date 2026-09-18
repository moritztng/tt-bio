#!/usr/bin/env bash
# c14-p300c-256-recell: record boltz2's p300c ladder with reps at every gated rung.
#
# Unattended by design: it may sit for an hour waiting for the box, then fold for ~16 min.
# Two guards, because a SUSPECT record here is worse than no record -- the cell it replaces is
# stale precisely because it was drawn on a busy box.
#   1. benchlock holds the box, with LOAD_WAIT set past its 900 s fall-through so it cannot
#      quietly proceed and print "treat this run as suspect".
#   2. record_inner.sh runs host_quiet.py INSIDE the lock, immediately before the record.
#      Not green -> no record, exit 3.
# loadavg and AICLK are sampled at 1 Hz for the whole run, so mid-run contention is visible
# afterwards rather than assumed absent from one snapshot at the start.
set -u
WT=/home/ttuser/.coworker/wt/c14-p300c-256-recell
cd "$WT" || exit 1
L=$WT/perf/c14_p300c_256/logs
mkdir -p "$L"
export CARD=1

( while :; do
    printf '%s %s %s\n' "$(date -u +%FT%TZ)" \
      "$(cat /sys/class/tenstorrent/tenstorrent\!"$CARD"/tt_aiclk 2>/dev/null)" \
      "$(cut -d' ' -f1 /proc/loadavg)"
    sleep 1
  done ) > "$L/aiclk_load_card${CARD}_boltz2.log" 2>/dev/null &
SAMPLER=$!
echo "sampler pid $SAMPLER (aiclk + loadavg, 1 Hz)"

export BENCHLOCK_WAIT_S=6000 BENCHLOCK_LOAD_WAIT_S=6000
timeout 7800 /home/ttuser/.coworker/scripts/benchlock.sh c14-p300c-256-recell -- \
    bash "$WT/perf/c14_p300c_256/record_inner.sh"
RC=$?
kill "$SAMPLER" 2>/dev/null
echo "RECORD_EXIT=$RC"
date -u +"finished %FT%TZ"
