#!/bin/bash
# Runs ON a rented GPU box: bare machine -> state/bc4/QUALITY-REF.json, unattended, inside a hard wall clock.
#
#   bash bc4_quality_session.sh --budget-min 150 [--concurrency 3] [--dry-run]
#
# The 2026-10-19 session capped mid-campaign and left 14 of 90 trajectories, usable by luck. This exists so the
# next top-up converts to a finished reference in ONE session with nobody watching:
#   * bc4_session_plan.py orders the campaigns so the gate-critical ones run first (the A/A control's missing arm,
#     then the second target). A session that runs out of budget still leaves the most useful campaigns on disk.
#   * --budget-min is a HARD wall. Every campaign is launched under `timeout` against the remaining budget, and
#     results are pushed back after every wave, not at the end, so a kill never costs more than one wave.
#   * The reference is built on the box from whatever landed, so the answer comes home even if nothing else does.
#
# Needs on the box: the tt-bio checkout at $WT (this script's repo) and the request JSON. Pushes results to
# $PUSH_TO (rsync target, default the fleet's artifacts path) after every wave.
set -uo pipefail

BUDGET_MIN=""; CONC=3; DRY=0
REQUEST=${REQUEST:-/root/kc/QUALITY-REF-REQUEST.json}
OUT=${OUT:-/root/kc/bc4-quality}
WT=${WT:-/root/tt-bio}
BC2=${BC2:-/root/bc2}
PUSH_TO=${PUSH_TO:-}
while [ $# -gt 0 ]; do
  case $1 in
    --budget-min) BUDGET_MIN=$2; shift 2;;
    --concurrency) CONC=$2; shift 2;;
    --request) REQUEST=$2; shift 2;;
    --out) OUT=$2; shift 2;;
    --push-to) PUSH_TO=$2; shift 2;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 2;;
  esac
done
[ -n "$BUDGET_MIN" ] || { echo "--budget-min is required: it is the hard wall, not a hint" >&2; exit 2; }

START=$(date +%s)
DEADLINE=$((START + ${BUDGET_MIN%.*} * 60))
mkdir -p "$OUT"
LOG=$OUT/session.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
left() { echo $((DEADLINE - $(date +%s))); }

say "budget ${BUDGET_MIN} min, concurrency $CONC, deadline $(date -u -d @$DEADLINE +%FT%TZ)"

# --- the plan ------------------------------------------------------------------------------------------------------
PLAN=$OUT/plan.json
python3 "$WT/perf/spd/bc4_session_plan.py" --request "$REQUEST" --budget-min "$BUDGET_MIN" \
  --concurrency "$CONC" --out "$PLAN" 2>&1 | tee -a "$LOG"
[ -s "$PLAN" ] || { say "FATAL: no plan"; exit 1; }

# --- setup ---------------------------------------------------------------------------------------------------------
# Idempotent and marker-guarded, so a relaunch after a crash does not reinstall. box_bc_setup.sh already carries the
# stock BindCraft 2 half (python 3.12, the pinned checkout, [cuda13], fetch-weights); bc2only skips the colabdesign
# half, which this reference does not use and which costs most of the setup time.
if [ $DRY -eq 0 ]; then
  if [ ! -f "$OUT/SETUP-OK" ]; then
    say "setup: stock BindCraft 2 at the pinned sha"
    mkdir -p /root/kc
    bash "$WT/perf/kitcmp/box_bc_setup.sh" bc2only >>"$LOG" 2>&1 \
      && touch "$OUT/SETUP-OK" || { say "FATAL: setup failed, see $LOG"; exit 1; }
  fi
  say "setup ok, $(left)s of budget left"
fi
PYBIN=${PYBIN:-/opt/bc2-venv/bin/python}

# --- the banked campaigns have to be ON the box before the reference is built --------------------------------------
# They are 14 real trajectories the 2026-10-19 B200 already paid for, and the request reads them into the reference
# without re-running them. A fresh box does not have them, so without this the session builds a reference missing
# a third of its pdl1 data and reports pdl1 short. Caught by the dry run, not by a real session, which is the point
# of having one.
if [ -n "${BANKED_FROM:-}" ]; then
  say "pulling the banked campaigns from $BANKED_FROM"
  while IFS=$'\t' read -r target arm seed; do
    d=$OUT/$target/${arm}_s${seed}
    [ -d "$d/campaign_s${seed}" ] && { say "  banked $target/${arm}_s${seed} already here"; continue; }
    mkdir -p "$d"
    rsync -az "$BANKED_FROM/$target/${arm}_s${seed}/" "$d/" >>"$LOG" 2>&1 \
      && say "  banked $target/${arm}_s${seed}" \
      || say "  WARNING: could not pull banked $target/${arm}_s${seed}; the reference will be short by its trajectories"
  done < <(python3 -c "
import json
for b in json.load(open('$REQUEST')).get('banked_runs', []):
    print('\t'.join((b['target'], b['arm'], str(b['campaign_seed']))))
")
fi

push() {
  [ -n "$PUSH_TO" ] || return 0
  # CSV and jsonl only: the .cif structures are large and nothing in the grade reads them.
  rsync -az --include='*/' --include='*.csv' --include='*.jsonl' --include='*.log' --include='*.json' \
        --exclude='*' "$OUT/" "$PUSH_TO/" >>"$LOG" 2>&1 \
    && say "pushed to $PUSH_TO" || say "WARNING: push to $PUSH_TO failed, results are still on the box at $OUT"
}

# --- the waves -----------------------------------------------------------------------------------------------------
WAVES=$(python3 -c "import json,sys; print(len(json.load(open('$PLAN'))['waves']))")
say "$WAVES waves scheduled"
for w in $(seq 0 $((WAVES - 1))); do
  REMAIN=$(left)
  SPAN=$(python3 -c "import json; print(json.load(open('$PLAN'))['waves'][$w]['span_s'])")
  if [ "$REMAIN" -le 120 ]; then say "wave $w skipped: ${REMAIN}s left"; break; fi
  # Run the wave against whatever budget is actually left, not against what the plan assumed. A wave that
  # overruns is capped rather than allowed to eat the build step at the end.
  CAP=$((REMAIN - 90))
  [ "$CAP" -gt "$SPAN" ] && CAP=$((SPAN + 300))
  say "wave $w: span ${SPAN}s, ${REMAIN}s left, capping each campaign at ${CAP}s"

  PIDS=""
  while IFS=$'\t' read -r id target arm seed example blen traj; do
    d=$OUT/$target/${arm}_s${seed}
    if [ -f "$d/rc" ]; then say "  $id already done, skipping"; continue; fi
    mkdir -p "$d"
    say "  launching $id -> $d"
    if [ $DRY -eq 1 ]; then echo 0 > "$d/rc"; continue; fi
    ( XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
      timeout "$CAP" "$PYBIN" "$WT/perf/kitcmp/bc2_gpu.py" \
        --bc2 "$BC2" --out "$d" --seed "$seed" --example "$example" \
        --binder-len "$blen" --max-traj "$traj" --final-designs 1000 \
        > "$d/stdout.log" 2> "$d/stderr.log"; echo $? > "$d/rc" ) &
    PIDS="$PIDS $!"
  done < <(python3 - "$PLAN" "$REQUEST" "$w" <<'PY'
import json, sys
plan, request, w = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), int(sys.argv[3])
by_id = {r["id"]: r for r in request["runs"]}
for rid in plan["waves"][w]["runs"]:
    r = by_id[rid]
    print("\t".join(str(x) for x in (r["id"], r["target"], r["arm"], r["campaign_seed"],
                                     r["example"], r["binder_len"], r["trajectories"])))
PY
  )
  for p in $PIDS; do wait "$p"; done
  say "wave $w done, $(left)s left"
  push
done

# --- the answer ------------------------------------------------------------------------------------------------------
# Always build, even on a short session: a partial reference that says what it is beats a directory of CSVs.
say "building the reference from whatever landed"
REF=$OUT/QUALITY-REF.json
python3 "$WT/perf/spd/bc2_quality.py" reference --request "$REQUEST" --root "$OUT" --out "$REF" \
  2>&1 | tee -a "$LOG"
GATE=$?
if [ $GATE -ne 0 ]; then
  say "the reference is SHORT of the gate; rebuilding it stamped provisional so nothing mistakes it for the real one"
  python3 "$WT/perf/spd/bc2_quality.py" reference --provisional --request "$REQUEST" --root "$OUT" \
    --out "$OUT/QUALITY-REF-PROVISIONAL.json" 2>&1 | tee -a "$LOG"
  rm -f "$REF"
fi
push
say "session finished after $(( ($(date +%s) - START) / 60 )) min; gate_rc=$GATE"
exit $GATE
