#!/bin/sh
# Chained p300c re-seed. Three stages on card 3, resumable across relaunches:
#   1. wait out the capacity sweep already running, fold its report into the baseline
#   2. re-measure the sweep's FAIL cells on the FIXED tree. The sweep process started at
#      01:42Z and loaded scripts/capacity_gate.py as it was then, before 8e3a3e9e added the
#      guard that a walk in which the allocator never refused is not a ceiling. Only a FAIL
#      cell runs a bisect, so only a FAIL cell can carry an invented ceiling note.
#   3. walk the size ladder one model at a time
# cwd is this worktree on purpose: a job rooted in another slug's worktree loses its files
# when fleet hygiene tears that worktree down.
#
# Completion markers are artifact-based, never exit codes: both gates exit nonzero when a
# cell legitimately FAILs, so an rc-keyed marker would re-run a finished model forever.
WT=/home/ttuser/.coworker/wt/capacity-size-ladder-reseed-p300c
OUT=$WT/perf/capacity-size-ladder-reseed-p300c
PY=/home/ttuser/tt-bio-dev/env/bin/python3
CAP_PID=570911
cd "$WT" || exit 1

log() { printf "%s %s\n" "$(date -u +%FT%TZ)" "$*" >> "$OUT/chain.log"; return 0; }

# Did model $1's capacity cell get re-measured into report $2?
have_cell() {
  $PY -c 'import json,sys
try: d=json.load(open(sys.argv[2]))
except Exception: sys.exit(1)
sys.exit(0 if any(r.get("model")==sys.argv[1] for r in d.get("results") or []) else 1)' "$1" "$2"
}

# Does the size-ladder baseline hold model $1 for p300c, at this commit, at every rung?
have_ladder() {
  $PY -c 'import json,subprocess,sys
m=sys.argv[1]
head=subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
try: d=json.load(open("docs/size_ladder_baseline.json"))
except Exception: sys.exit(1)
e=(d.get("cards") or {}).get("p300c",{}).get("models",{}).get(m)
if not e or e.get("commit")!=head: sys.exit(1)
want={256,512,640,768,896,1024}|({1088} if m=="rf3" else set())
got={int(r) for r in (e.get("runtime_s") or {})}
sys.exit(0 if want<=got else 1)' "$1"
}

log "chain v2 start, waiting on capacity sweep pid $CAP_PID"
while kill -0 "$CAP_PID" 2>/dev/null; do sleep 30; done
log "capacity sweep exited"

if [ ! -f "$OUT/cap_recorded" ]; then
  PYTHONPATH=$WT $PY scripts/capacity_gate.py --record-from "$OUT/capgate_p300c.json" \
    >> "$OUT/chain.log" 2>&1
  log "capacity --record-from rc=$?"
  : > "$OUT/cap_recorded"
fi

# Re-measure only the FAIL cells whose note 8e3a3e9e's guard would actually change: the ones
# where no bisect rung saw the allocator refuse. A cell that already carries a real refusal gets
# the identical note from the fixed code, so re-measuring it reproduces its own result at full
# cost, and opendde bisects at the RESIDENCY tier at roughly 12 min a rung. See
# needs_remeasure.py. --record is partial, so each run updates only its own cell.
FAILS=$($PY "$OUT/needs_remeasure.py" "$OUT/capgate_p300c.json")
log "FAIL cells the guard would change, so re-measuring: ${FAILS:-none}"
for m in $FAILS; do
  if have_cell "$m" "$OUT/capgate_refail_$m.json"; then
    log "re-measure $m already has a cell, skipping"; continue
  fi
  log "re-measure $m start"
  TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
  PYTHONPATH=$WT $PY scripts/capacity_gate.py --workers tt-quietbox2:3 --models "$m" --record \
    --work-dir "$OUT/work" --report "$OUT/capgate_refail_$m.json" > "$OUT/refail_$m.log" 2>&1
  log "re-measure $m rc=$?"
done

for m in esmfold2 boltz2 protenix-v1 protenix-v2 opendde nesso1 openfold3 openbind rf3; do
  if have_ladder "$m"; then log "size-ladder $m already recorded, skipping"; continue; fi
  log "size-ladder $m start"
  TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 \
  TT_BIO_LEASE_HOLDER=worker:capacity-size-ladder-reseed-p300c \
  PYTHONPATH=$WT $PY scripts/release_gate.py --model size-ladder \
    --size-ladder-record --size-ladder-models "$m" > "$OUT/ladder_$m.log" 2>&1
  log "size-ladder $m rc=$?"
  if have_ladder "$m"; then log "size-ladder $m recorded"; else log "size-ladder $m NOT recorded"; fi
done
log "chain complete"
