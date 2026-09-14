#!/usr/bin/env bash
# The release gate on the default-ON tree, one arm at a time on card 3, resumable.
#
# Arm order is by what can still change the verdict. The lever is gated strictly above 1024
# tokens, so the arms that can see it at all come first: pytest (the shipping-default
# assertion), the size ladder (rf3 reaches 1088) and the capacity gate (1536). The 44-leg
# accuracy gate is hours and every one of its targets is below the cap, so it runs as the
# neutrality control rather than as the discriminator.
#
# Timed arms wait for a quiet box; correctness arms only lose wall-clock to load, so they do
# not. Every arm appends one line to progress and is skipped if that line is already there.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate"
PROG="$OUT/progress"
CARD=3
P=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-fused-sdpa-default-ship

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

# The quiet ladder owns card 3 until it is done; two drivers on one card is a wedge. Wait on its
# own end marker plus a live fold, not on a pgrep of the driver name: the launching `bash -c`
# carries the script name in its argv and outlives it, so a name match alone never clears.
QL=perf/ttx_a3/nochange/quiet/driver.log
waited=0
while [ "$waited" -lt 5400 ]; do
  grep -q 'QUIET DONE' "$QL" && ! pgrep -f fold_parity_a3.py > /dev/null && break
  sleep 30; waited=$((waited + 30))
done
log "quiet-ladder wait ended after ${waited}s"

wait_quiet() {  # $1 = max seconds to wait, $2 = loadavg ceiling
  local waited=0
  while [ "$waited" -lt "$1" ]; do
    awk -v c="$2" '{exit !($1 < c)}' /proc/loadavg && return 0
    sleep 60; waited=$((waited + 60))
  done
  return 1
}

run_arm() {  # $1 = name, $2 = needs a quiet box (0/1), rest = argv
  local name="$1" timed="$2"; shift 2
  # Only a recorded rc= counts as done: an arm killed by a reboot has a START line and must rerun.
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  if [ "$timed" = 1 ]; then
    if wait_quiet 10800 8.0; then log "$name QUIET-OK loadavg=$(cut -d' ' -f1 /proc/loadavg)"
    else log "$name SKIPPED-LOADED loadavg=$(cut -d' ' -f1 /proc/loadavg) after 3h wait"; return 0; fi
  fi
  log "$name START loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  TT_VISIBLE_DEVICES=$CARD timeout 43200 "$@" > "$OUT/$name.log" 2>&1
  log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
}

# The suite opens a device, so it is pinned rather than run card-free.
run_arm pytest        0 $P -m pytest -q --tb=short
run_arm size-ladder   1 $P scripts/release_gate.py --model size-ladder
run_arm capacity      0 $P scripts/capacity_gate.py
run_arm ux            0 $P scripts/ux_regression.py
run_arm parity        0 $P scripts/full_parity_gate.py --workers tt-quietbox2:$CARD \
          --workdir "$OUT/gate-8d40acab6" --out "$OUT/parity.json"
run_arm perf          1 $P scripts/perf_regression.py
log "GATE_DRIVER_DONE"
