#!/usr/bin/env bash
# The whole gate as ONE serial chain, because two device lanes reset this box.
#
# Three wedges and a host reset in 45 minutes, all of them at loadavg 4-9 with a second ladder
# lane and a pytest chunk running; the same arms at loadavg 0.04-0.46 completed. qb2's watchdog
# fires more often under load, so the gate now runs one arm at a time and takes longer on purpose.
#
# Phase order is by what can still change the verdict per minute spent: the cheap correctness arms
# first, then the ladder (the only arm that is genuinely new information on this tree), then
# capacity, then the timed arms. Every arm is skip-guarded by its `rc=` line in the run's progress
# file, so a reset costs one arm and a re-run of this script costs nothing.
set -u
WT="${GATE_WT:-/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge2}"
cd "$WT" || exit 1
RUN="${GATE_OUT:-perf/ttx_a3/gate6}"
CARD="${GATE_CARD:-0}"
HOLDER="${GATE_HOLDER:-worker:ttx-a3-sdpa-ship-remerge2}"
WORKER="${GATE_WORKER:-tt-quietbox2}"
PROG="$WT/$RUN/progress"
mkdir -p "$WT/$RUN"; touch "$PROG"
log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }
export GATE_WT="$WT" GATE_OUT="$RUN" GATE_CARD="$CARD" GATE_HOLDER="$HOLDER" GATE_WORKER="$WORKER"
export GATE_SKIP_NEUT=1

# The wedge healer, one per box: a fold stuck at `trunk 0/4` holds the card and burns no CPU, and
# without this the lane waits out its own 3600 s timeout. Started here so a boot relaunch gets it
# back too.
pgrep -f '^bash perf/ttx_a3/wedge_watch\.sh$' > /dev/null || \
  setsid nohup bash perf/ttx_a3/wedge_watch.sh >> "$WT/$RUN/wedge_watch.log" 2>&1 < /dev/null &

phase() { log "PHASE $1"; }

phase "1/4 correctness: pytest, the flag-off control, ux"
GATE_SKIP_CAP=1 GATE_SKIP_LADDER=1 GATE_SKIP_PERF=1 GATE_SKIP_PARITY=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

phase "2/4 size ladder: splice the new counter into every model, then check"
CARD="$CARD" bash perf/ttx_a3/ladder_campaign.sh \
  boltz2 rf3 esmfold2 protenix-v2 openfold3 opendde nesso1 openbind protenix-v1 \
  >> "$WT/$RUN/campaign.log" 2>&1

phase "3/4 capacity, 15 cells"
GATE_SKIP_LADDER=1 GATE_SKIP_PERF=1 GATE_SKIP_PARITY=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

phase "4/4 perf under benchlock, then the 44-leg parity gate"
GATE_SKIP_CAP=1 GATE_SKIP_LADDER=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

log "SERIAL_GATE_DONE"
