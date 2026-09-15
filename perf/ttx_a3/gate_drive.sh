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

# qb2's mean uptime is 52 min over its last 14 boots (731 min, 2026-09-14T15:07Z ->
# 2026-09-15T03:18Z). An arm that needs longer than one boot never records an rc and so reruns
# from scratch forever: the monolithic capacity roster is ~2 h, the size ladder is 9 models x 4
# rungs and the perf arm is 20 models. All three take a model list, so all three are chunked one
# model per arm and each chunk fits inside a boot.
#
# Chunking is not a weaker check. A capacity cell is per (model, card type) and independent of
# every other cell. release_gate.py's own --size-ladder-models exists for this exact reason ("a
# 6-model record is ~2 h of device time, so it has to be resumable a model at a time"); the only
# thing a per-model ladder run skips is _size_ladder_coverage_gap, which
# tests/test_size_ladder_gate.py checks in the pytest arm already. perf_regression.py --model is
# repeatable and scores per model against docs/perf_baselines.json either way.
#
# --no-card-reset is not optional on this box: a tt-smi reset takes the whole board PAIR down and
# sibling campaigns hold the other cards.
BENCH=/home/ttuser/.coworker/scripts/benchlock.sh

CAP_MODELS="boltz2 esmfold2 esmfold2-fast protenix-v1 protenix-v2 openfold3 openbind opendde \
            opendde-abag rf3 esmc-300m esmc-600m esmc-6b saprot-35m saprot-650m"
LADDER_MODELS="boltz2 esmfold2 protenix-v1 protenix-v2 openfold3 opendde rf3 nesso1 openbind"
PERF_MODELS="boltz2 boltz2-affinity esmfold2 esmfold2-fast protenix-v1 protenix-v2 openfold3 \
             openbind opendde opendde-abag rf3 rfd3 pxdesign boltzgen nesso1 esmc-300m \
             esmc-300m-single esmc-600m esmc-6b saprot-650m"

# ux is 10 minutes and it holds the last three of the eight reds the weights outage caused, so it
# runs before the long arms.
run_arm ux 0 $P scripts/ux_regression.py

# boltz2 leads the roster: 1536 tokens is the only gate arm that folds in this lever's own regime,
# and boltz2 is the model the 1.1856x was measured on.
for m in $CAP_MODELS; do
  run_arm "capacity-$m" 0 $P scripts/capacity_gate.py --models "$m" \
      --workers "tt-quietbox2:$CARD" --no-card-reset \
      --work-dir "$OUT/cap-$m" --report "$OUT/capacity_$m.json"
done

# Untimed. The ladder's verdict is the fired/dark lever census plus a runtime exponent whose
# tolerance floors at +-0.50; co-tenant noise on this box is 1-10 % and a load factor common to
# two rungs cancels out of their ratio. Holding benchlock across 9 models x 4 rungs would park
# every other perf task on qb2 for hours to tighten a band that is already 5x the signal.
for m in $LADDER_MODELS; do
  run_arm "ladder-$m" 0 $P scripts/release_gate.py --model size-ladder --size-ladder-models "$m"
done

# perf_regression compares wall clock against docs/perf_baselines.json, so it is the one arm where
# a co-tenant makes the number wrong rather than slow. It takes benchlock, which also excludes a
# foreign fold instead of merely sampling loadavg.
for m in $PERF_MODELS; do
  run_arm "perf-$m" 0 bash $BENCH "$HOLDER" -- $P scripts/perf_regression.py --model "$m"
done

# gate2's parity arm reported 28 of 44 legs ERROR and it was not an accuracy result: one leg
# leaked a live process still holding card 0's flock, so every leg after it waited the lease's
# 120 s and died with DeviceInUseError. The message even said "the same holder identity in a
# DIFFERENT process, so this is a real co-tenant, not a stale lease" -- correct, and the reason
# 28 tracebacks had to be read before the arm's rc=1 meant anything. Name the holders BEFORE the
# arm starts so a pre-existing leak is a one-line refusal instead of an hour of timeouts. This
# only catches a leak that predates the arm; one that happens mid-arm belongs to
# full_parity_gate.py's leg reaping, which is shared code and may not change while this resumes.
parity_card_holders() {
  local pids
  pids=$(lsof -t "/dev/tenstorrent/$CARD" 2>/dev/null | tr '\n' ' ')
  printf '%s' "$pids"
}
if ! grep -q " parity rc=" "$PROG" 2>/dev/null; then
  h=$(parity_card_holders)
  if [ -n "$h" ]; then
    log "parity PRECHECK card $CARD already held by pids: $h -- $(ps -o pid=,cmd= -p ${h% } 2>/dev/null | tr '\n' ';')"
  else
    log "parity PRECHECK card $CARD free, 0 holders"
  fi
fi

# Last, and the only arm that resumes on its own: full_parity_gate caches per-leg verdicts in the
# workdir and fingerprints tt_bio/ + scripts/, so nothing under tt_bio/ or scripts/ may change
# once this starts. All 44 legs are below the 1024-token cap, so this is the neutrality control
# for the fallthrough, not evidence about the route.
run_arm parity        0 $P scripts/full_parity_gate.py --workers tt-quietbox2:$CARD \
          --workdir "$OUT/parity-workdir" --out "$OUT/parity.json"
log "GATE_DRIVER_DONE"
