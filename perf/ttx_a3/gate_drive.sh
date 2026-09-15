#!/usr/bin/env bash
# The release gate on the default-ON tree, one arm at a time on the granted card, resumable.
#
# Arm order is by what can still change the verdict. The lever is gated strictly above 1024
# tokens, so the arms that can see it at all come first: pytest (the shipping-default
# assertion), the size ladder (rf3 reaches 1088) and the capacity gate (1536). The 44-leg
# accuracy gate is hours and every one of its targets is below the cap, so it runs as the
# neutrality control rather than as the discriminator.
#
# The below-cap ladder now runs on a sibling card instead of this one, so the correctness arms
# no longer wait it out. The TIMED arms still do: a fold on another card is another 100 % core
# on the same 16-core host, and these two arms are compared against a recorded baseline.
# Every arm appends one line to progress and is skipped if that line is already there.
set -u
# Worktree, card and output dir are inputs, not constants: this gate has now run from two
# worktrees on two card grants, and a hardcoded path silently writes another worker's tree.
# GATE_OUT names the run, so a re-gate on a re-merged tree starts on a fresh progress file
# instead of skipping every arm the previous tree already recorded.
WT="${GATE_WT:-/home/ttuser/.coworker/wt/ttx-a3-fused-sdpa-default-ship}"
cd "$WT" || exit 1
OUT="$WT/${GATE_OUT:-perf/ttx_a3/gate}"
PROG="$OUT/progress"
CARD="${GATE_CARD:-1}"
HOLDER="${GATE_HOLDER:-worker:ttx-a3-fused-sdpa-default-ship}"
P=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=$HOLDER

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

QL=perf/ttx_a3/nochange/quiet/driver.log

# A timed arm needs the box to itself. Two conditions, both required: the sibling-card ladder is
# finished (its own end marker AND no live fold -- the launching `bash -c` keeps the script name
# in its argv and outlives it, so a name match alone never clears), and loadavg is under the
# ceiling so a sibling campaign that relaunched has not taken the window back.
wait_quiet() {  # $1 = max seconds to wait, $2 = loadavg ceiling
  local waited=0
  while [ "$waited" -lt "$1" ]; do
    # Gate on a LIVE fold, not on the ladder's end marker. The marker only prints once all four
    # rungs are done and the ladder has never got that far, so requiring it parked both timed arms
    # for the full 3h and then skipped them -- the opposite of waiting for a quiet box.
    if ! pgrep -f fold_parity_a3.py > /dev/null; then
      awk -v c="$2" '{exit !($1 < c)}' /proc/loadavg && return 0
    fi
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

# Untimed arms first, and the hours-long accuracy arm last. qb2 watchdog-reset three times in
# the 90 minutes this gate has been trying to run, always at loadavg 15+ while four campaigns
# share 16 cores, so an arm that needs more than about half an hour of uninterrupted box is a
# coin flip. Ordering by "can this arm still finish" beats ordering by "can this arm see the
# lever": a timed arm parked in wait_quiet blocks every untimed arm behind it for up to 3h.
#
# The suite opens a device, so it is pinned rather than run card-free. `-rf` because the run that
# died at 89 % had nine F marks and no summary line, which names nothing.
run_arm pytest        0 $P -m pytest -q --tb=line -rf
run_arm capacity      0 $P scripts/capacity_gate.py
run_arm ux            0 $P scripts/ux_regression.py
run_arm size-ladder   1 $P scripts/release_gate.py --model size-ladder
# perf_regression compares wall clock against docs/perf_baselines.json, so it is the one arm
# where a co-tenant turns a slow measurement into a wrong one. It takes benchlock rather than
# wait_quiet: benchlock also excludes a foreign fold, and it interlocks with the sibling perf
# campaigns on this box instead of merely sampling loadavg. The size ladder stays on wait_quiet
# -- its exponent tolerance is +-0.50, far above co-tenant noise, and an exclusive hold across
# 9 models x 4 rungs would park every other perf task on qb2 for hours.
BENCH=/home/ttuser/.coworker/scripts/benchlock.sh
run_arm perf          0 bash $BENCH "$HOLDER" -- $P scripts/perf_regression.py
run_arm parity        0 $P scripts/full_parity_gate.py --workers tt-quietbox2:$CARD \
          --workdir "$OUT/parity-workdir" --out "$OUT/parity.json"
log "GATE_DRIVER_DONE"
