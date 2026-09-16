#!/usr/bin/env bash
# The whole gate as ONE serial chain, because two device lanes reset this box.
#
# Three wedges and a host reset in 45 minutes, all of them at loadavg 4-9 with a second ladder
# lane and a pytest chunk running; the same arms at loadavg 0.04-0.46 completed. qb2's watchdog
# fires more often under load, so the gate now runs one arm at a time and takes longer on purpose.
#
# Phase order is by what can still change the verdict per minute spent, and that order was WRONG
# for four passes. `SIZE_LADDER_RUNGS` is 256..1024 and the lever fires on `q_len >
# _Q_SPLIT_MAX_S` = 1024 (`tt_bio/triatt_sdpa.py:88`), so a shared rung can never reach it: of the
# 9 models this arm walks, `SIZE_LADDER_EXTRA_RUNGS` gives exactly one cell above the cap, rf3 at
# 1088. Walking boltz2's six blind rungs first spent ~6 folds before touching the only ladder cell
# that can discriminate, on a box that wedges about 1 fold in 6 -- which is why five passes
# recorded no verdict. In-regime arms now go first: rf3, then capacity (it tests at the 1536 aa
# ceiling), then the timed perf and parity arms. The other 8 ladder models are a release-gate
# completeness tail, not lever evidence, so they run last.
# Every arm is skip-guarded by its `rc=` line in the run's progress file, so a reset costs one arm
# and a re-run of this script costs nothing.
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

phase "1/5 correctness: pytest, the flag-off control, ux"
GATE_SKIP_CAP=1 GATE_SKIP_LADDER=1 GATE_SKIP_PERF=1 GATE_SKIP_PARITY=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

# rf3 alone: its 1088 rung is the ONLY ladder cell above `_Q_SPLIT_MAX_S`, so this is the only
# part of the ladder arm that carries information about the lever.
phase "2/5 size ladder, in-regime: rf3 (the 1088 rung, the one cell above the cap)"
CARD="$CARD" bash perf/ttx_a3/ladder_campaign.sh rf3 \
  >> "$WT/$RUN/campaign.log" 2>&1

phase "3/5 capacity, 15 cells (tests at the 1536 aa ceiling, in the lever's regime)"
GATE_SKIP_LADDER=1 GATE_SKIP_PERF=1 GATE_SKIP_PARITY=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

phase "4/5 perf under benchlock, then the 44-leg parity gate"
GATE_SKIP_CAP=1 GATE_SKIP_LADDER=1 \
  bash perf/ttx_a3/gate_drive.sh >> "$WT/$RUN/driver.log" 2>&1

# Below the cap by construction, so these 8 models can only ever reproduce main's own state. The
# release gate still wants them green; this lever's verdict does not wait on them.
phase "5/5 size ladder, remaining 8 models: release-gate completeness, blind to the lever"
CARD="$CARD" bash perf/ttx_a3/ladder_campaign.sh \
  boltz2 esmfold2 protenix-v2 openfold3 opendde nesso1 openbind protenix-v1 \
  >> "$WT/$RUN/campaign.log" 2>&1

log "SERIAL_GATE_DONE"
