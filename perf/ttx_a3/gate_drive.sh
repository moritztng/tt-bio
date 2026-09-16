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
# The worker host is an input too. This gate was written for qb2, and on 2026-09-15 qb2 was
# reachable and unreachable inside the same ten-minute span -- the watchdog cadence, not a dead
# box -- so the long arms moved to qb1's p150a cards. A hardcoded "tt-quietbox2" here silently
# points capacity_gate and full_parity_gate at the wrong host while every other path in this
# script is already host-agnostic.
WORKER="${GATE_WORKER:-tt-quietbox2}"
HOLDER="${GATE_HOLDER:-worker:ttx-a3-fused-sdpa-default-ship}"
P=/home/ttuser/tt-bio-dev/env/bin/python3
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export OF3_CKPT=/home/ttuser/.boltz/of3-p2-155k.pt
export ESM_ROOT=/home/ttuser/esm
# The grant is an input too: a fan-out onto an idle sibling card runs on CARD but must
# declare the widened grant, or tt-bio refuses the open.
export TT_BIO_LEASE_CARDS=${GATE_LEASE_CARDS:-$CARD}
export TT_BIO_LEASE_HOLDER=$HOLDER

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

# CARD is a UMD logical id -- it is what TT_VISIBLE_DEVICES, TT_BIO_LEASE_CARDS and fleet.sh all
# speak -- and it is NOT the /dev/tenstorrent/N node number. UMD numbers its chips by sorting the
# PCI devices by BDF; the kernel driver numbers its nodes in probe order, and the two only agree
# by luck. On qb2 they happened to agree, which is why every `lsof` on a node path in this script
# looked correct for three passes. On qb1 they are a rotation:
#
#   UMD 0 = 0000:01 = node 1     UMD 2 = 0000:42 = node 3
#   UMD 1 = 0000:41 = node 2     UMD 3 = 0000:c1 = node 0
#
# verified by opening each TT_VISIBLE_DEVICES=K in turn and reading lsof. Indexing a node path
# with a UMD id there does not merely look at the wrong card, it makes reap_card send SIGKILL to
# whatever co-tenant holds it: a gate granted UMD 3 would have reaped node 3, which is UMD 2.
card_node() {  # UMD logical id -> /dev/tenstorrent/N, by sorting nodes on BDF exactly as UMD does
  local want="$1" i=0 n
  for n in $(for d in /sys/class/tenstorrent/tenstorrent!*; do
               [ -e "$d/device/uevent" ] || continue
               printf '%s %s\n' "$(sed -n 's/^PCI_SLOT_NAME=//p' "$d/device/uevent")" "${d##*!}"
             done | sort | cut -d' ' -f2); do
    [ "$i" = "$want" ] && { printf '%s' "$n"; return 0; }
    i=$((i + 1))
  done
  return 1
}
NODE="$(card_node "$CARD")" || { echo "cannot resolve UMD card $CARD to a device node" >&2; exit 1; }
log "CARD=$CARD (UMD) resolves to /dev/tenstorrent/$NODE"

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
  # ARM_TIMEOUT, because 43200 is right for a 15-model capacity roster and catastrophic for a
  # single fold. The 768 aa rung wedges rather than failing -- gate3's first attempt sat 10
  # minutes after a caught L1 throw with no further output -- and at the default it would have
  # held the whole re-gate for 12 hours on one known-wedging rung.
  TT_VISIBLE_DEVICES=$CARD timeout -k 30 "${ARM_TIMEOUT:-43200}" "$@" > "$OUT/$name.log" 2>&1
  log "$name rc=$? loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  reap_card "$name"
}

# `timeout` signals only its direct child, so a fold that forked a worker leaves the worker alive
# and holding the card. That is not a hypothetical: it is precisely what turned gate2's parity arm
# into 28 consecutive DeviceInUseError legs, each burning the lease's 120 s wait behind one leaked
# process. After an arm has exited, anything still holding this card is leaked by definition, so
# reap it by EXPLICIT pid from lsof -- never a pkill pattern, which on this box would also match a
# sibling campaign's fold on another card.
reap_card() {
  local pids p
  pids=$(lsof -t "/dev/tenstorrent/$NODE" 2>/dev/null | tr '\n' ' ')
  [ -z "$pids" ] && return 0
  log "REAP after $1: card $CARD still held by$(for p in $pids; do printf ' %s(%s)' "$p" "$(ps -o comm= -p "$p" 2>/dev/null)"; done)"
  for p in $pids; do kill -TERM "$p" 2>/dev/null; done
  sleep 10
  for p in $(lsof -t "/dev/tenstorrent/$NODE" 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
}

# Untimed arms first, and the hours-long accuracy arm last. qb2 watchdog-reset three times in
# the 90 minutes this gate has been trying to run, always at loadavg 15+ while four campaigns
# share 16 cores, so an arm that needs more than about half an hour of uninterrupted box is a
# coin flip. Ordering by "can this arm still finish" beats ordering by "can this arm see the
# lever": a timed arm parked in wait_quiet blocks every untimed arm behind it for up to 3h.
#
# FIRST, because they are the only cheap arms that discriminate, and because the re-merge owes
# them. 768 aa is the one size below the cap whose on-arm never completed: gate2 got one off-arm
# fold at 47.2 s and then three timeouts, which main has since root-caused to this box's watchdog
# rather than to L1 or to this lever. 1024 aa is the cap boundary AND the shape main's newly
# default-on TT_BIO_TRANSITION_L1_ROWS moves, so the merged tree has to be read as a stack: two
# levers that each claim "bit-identical below the cap" have never been measured together.
#
# off/on/off, one arm PER PROCESS. Flipping the flag inside one live device context is the harness
# trap this campaign already fell into once, and the hang it manufactured was mis-attributed to
# the arm switch for a full pass before single-arm processes reproduced it.
# 500 s each: 768 aa completed in 47.196 s and 1024 aa in 57.775-59.067 s when they completed,
# so this is 8x the longest good fold and still lets all six arms fit inside one of this box's
# boots. An arm that hits it records rc=124 and the next arm runs.
# GATE_SKIP_NEUT names, honestly, that these two sizes are already recorded somewhere else rather
# than pretending they ran here. gate4 on qb1 folded both off/on/off with one arm per process and
# got byte-identical CIFs (768 aa 38aabd4058facb3f, 1024 aa 649aad7b46727c7e), so re-running them
# on a box that has never once completed the 768 aa on-arm spends windows to re-learn a banked
# result. Do NOT write fake rc= lines to get the same effect: the skip has to be visible in the
# progress file as a skip, or a later reader cannot tell measured from assumed.
ARM_TIMEOUT=500
if [ "${GATE_SKIP_NEUT:-0}" = 1 ]; then
  log "neut768/neut1024 SKIPPED-BANKED-ELSEWHERE: byte-identical off/on/off in gate4 on qb1"
else
for sz in 768 1024; do
  for arm in off1:off on1:on off2:off; do
    tag="${arm%%:*}"; a="${arm##*:}"
    run_arm "neut$sz-$tag" 0 $P perf/ttx_a3/fold_parity_a3.py \
        --dir "$OUT/f$sz" --arm "$a" --tag "cdk2x2_${sz}_$tag" --fixture "cdk2x2_$sz"
  done
done
fi

unset ARM_TIMEOUT

# The suite opens a device, so it is pinned rather than run card-free. `-rf` because the run that
# died at 89 % had nine F marks and no summary line, which names nothing.
# The suite is CHUNKED, for the same reason the model rosters are: an arm that cannot finish inside
# one boot never records an rc, so it restarts from scratch forever and the gate livelocks while
# looking busy. qb2's last seven boot intervals were 18.5, 15.9, 15.6, 14.7, 9.3, 30.0 and 18.8
# minutes, a 17.5 min mean, and the whole suite is about 20 minutes. That is not "slow", it is
# arithmetically unable to complete: every relaunch got about as far as 52 % and died.
#
# Round-robin over the SORTED file list, not a hand-written list of files. A static list silently
# stops covering whatever test file is added next, which is a recurring defect class here; this way
# a new tests/test_*.py joins a chunk the first time the gate runs after it lands. Six chunks puts
# each at roughly three minutes, comfortably inside even a bad window.
#
# Chunking is not a weaker check: every file lands in exactly one chunk, so the same tests run. The
# cost is re-paying pytest startup and the ttnn import per chunk.
PYTEST_CHUNKS=6
for k in $(seq 0 $((PYTEST_CHUNKS - 1))); do
  files=$(ls tests/test_*.py | sort | awk -v k="$k" -v n="$PYTEST_CHUNKS" 'NR % n == k')
  [ -z "$files" ] && continue
  run_arm "pytest-$k" 0 $P -m pytest -q --tb=line -rf $files
done

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

# THE CONTROL, and it runs here rather than at the end because it is the only arm that can turn a
# red into an attributed red. Same tree, same card, same chunks, one env var different. A suite
# failure means nothing on its own -- this tree carries 66 commits of other people's work -- so the
# question is never "did anything fail" but "did anything fail that does NOT fail with the flag off".
#
# The control self-validates, which is the part that is easy to leave out and fatal to leave out: a
# no-op control produces a reassuring empty only-in-ON set and means nothing at all.
# tests/test_sdpa_fused_pairs.py asserts the flag reads True, so it MUST fail in the control and
# only in the control. If pytestoff's failure list does not contain
# test_the_above_cap_route_is_strictly_above_the_cap, the control did not actually take effect and
# its verdict is void.
for k in $(seq 0 $((PYTEST_CHUNKS - 1))); do
  files=$(ls tests/test_*.py | sort | awk -v k="$k" -v n="$PYTEST_CHUNKS" 'NR % n == k')
  [ -z "$files" ] && continue
  run_arm "pytestoff-$k" 0 env TT_BIO_SDPA_FUSED_LARGE_S=0 $P -m pytest -q --tb=line -rf $files
done

# ux is 10 minutes and it holds the last three of the eight reds the weights outage caused, so it
# runs before the long arms.
run_arm ux 0 $P scripts/ux_regression.py

# boltz2 leads the roster: 1536 tokens is the only gate arm that folds in this lever's own regime,
# and boltz2 is the model the 1.1856x was measured on.
if [ "${GATE_SKIP_CAP:-0}" = 1 ]; then log "capacity SKIPPED by GATE_SKIP_CAP"; else
for m in $CAP_MODELS; do
  run_arm "capacity-$m" 0 $P scripts/capacity_gate.py --models "$m" \
      --workers "$WORKER:$CARD" --no-card-reset \
      --work-dir "$OUT/cap-$m" --report "$OUT/capacity_$m.json"
done
fi

# Untimed. The ladder's verdict is the fired/dark lever census plus a runtime exponent whose
# tolerance floors at +-0.50; co-tenant noise on this box is 1-10 % and a load factor common to
# two rungs cancels out of their ratio. Holding benchlock across 9 models x 4 rungs would park
# every other perf task on qb2 for hours to tighten a band that is already 5x the signal.
if [ "${GATE_SKIP_LADDER:-0}" = 1 ]; then log "size-ladder SKIPPED by GATE_SKIP_LADDER"; else
for m in $LADDER_MODELS; do
  run_arm "ladder-$m" 0 $P scripts/release_gate.py --model size-ladder --size-ladder-models "$m"
done
fi

# perf_regression compares wall clock against docs/perf_baselines.json, so it is the one arm where
# a co-tenant makes the number wrong rather than slow. It takes benchlock, which also excludes a
# foreign fold instead of merely sampling loadavg.
if [ "${GATE_SKIP_PERF:-0}" = 1 ]; then log "perf SKIPPED by GATE_SKIP_PERF"; else
for m in $PERF_MODELS; do
  run_arm "perf-$m" 0 bash $BENCH "$HOLDER" -- $P scripts/perf_regression.py --model "$m"
done
fi

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
  pids=$(lsof -t "/dev/tenstorrent/$NODE" 2>/dev/null | tr '\n' ' ')
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
if [ "${GATE_SKIP_PARITY:-0}" = 1 ]; then log "parity SKIPPED by GATE_SKIP_PARITY"; else
run_arm parity        0 $P scripts/full_parity_gate.py --workers "$WORKER:$CARD" \
          --workdir "$OUT/parity-workdir" --out "$OUT/parity.json"
fi
log "GATE_DRIVER_DONE"
