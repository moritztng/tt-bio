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
# One wave at a time, re-planned against the rate THIS card delivers on THIS target.
#
# The plan prices a trajectory from a constant, and the constant has always come from a different run than the one
# being planned. On 2026-10-19 that constant was a B200's 85.2 s while the card in hand delivered 194.4 s, and the
# modelled rate decides how many waves the budget buys: too slow a number drops waves the box had time for, too
# fast a number caps each campaign at a span it cannot finish in and truncates every one of them. Neither error is
# visible until the session is over and the money is spent.
# A target the fleet has never run has no constant at all. hIL2R is 387 residues against hPDL1's 115, so no pdl1
# measurement predicts it to better than a factor of several. So: run the first wave on the entering estimate,
# then take the delivered per-trajectory time out of runs.jsonl and re-plan what is left with it. By the second
# wave the schedule is priced on the card and the target actually in front of it.
rate_from_runs() {   # median seconds between consecutive trajectories in the campaigns just run, per stream
  local conc=$1; shift
  python3 - "$conc" "$@" <<'PY'
import json, statistics, sys
conc = int(sys.argv[1]); d = []
for p in sys.argv[2:]:
    try:
        ts = [json.loads(l)["t"] for l in open(p) if json.loads(l).get("ev") == "rep"]
    except OSError:
        continue
    d += [b - a for a, b in zip(ts, ts[1:])]
print(round(statistics.median(d) / max(conc, 1), 1) if d else "")
PY
}

w=-1
while :; do
  w=$((w + 1))
  REMAIN=$(left)
  if [ "$REMAIN" -le 120 ]; then say "wave $w skipped: ${REMAIN}s left"; break; fi
  # Re-plan from the campaigns that have NOT run yet, against the budget that is actually left. A campaign with
  # an rc is finished or partial, and either way is never re-run (a repeated seed cannot pair), so dropping it
  # from the request is what lets the next wave see the budget its predecessors did not spend.
  RREQ=$OUT/request-remaining.json
  python3 - "$REQUEST" "$OUT" "$RREQ" <<'PY'
import json, os, sys
req = json.load(open(sys.argv[1])); out = sys.argv[2]
req["runs"] = [r for r in req.get("runs", [])
               if not os.path.exists(os.path.join(out, r["target"], f"{r['arm']}_s{r['campaign_seed']}", "rc"))]
json.dump(req, open(sys.argv[3], "w"), indent=1)
PY
  [ "$(python3 -c "import json;print(len(json.load(open('$RREQ'))['runs']))")" -gt 0 ] || { say "no campaigns left to run"; break; }
  BC4_S_PER_TRAJ=${BC4_S_PER_TRAJ:-} BC4_SETUP_S=0 \
    python3 "$WT/perf/spd/bc4_session_plan.py" --request "$RREQ" --budget-min "$(python3 -c "print($REMAIN/60)")" \
      --concurrency "$CONC" --out "$PLAN" >>"$LOG" 2>&1
  [ -s "$PLAN" ] || { say "FATAL: no plan for wave $w"; break; }
  if [ "$(python3 -c "import json;print(len(json.load(open('$PLAN'))['waves']))")" -eq 0 ]; then
    say "wave $w: ${REMAIN}s left buys no further campaign at ${BC4_S_PER_TRAJ:-default} s/trajectory"; break
  fi
  SPAN=$(python3 -c "import json; print(json.load(open('$PLAN'))['waves'][0]['span_s'])")
  # Run the wave against whatever budget is actually left, not against what the plan assumed. A wave that
  # overruns is capped rather than allowed to eat the build step at the end.
  CAP=$((REMAIN - 90))
  [ "$CAP" -gt "$SPAN" ] && CAP=$((SPAN + 300))
  say "wave $w: span ${SPAN}s, ${REMAIN}s left, capping each campaign at ${CAP}s"

  PIDS=""; LAUNCHED=0
  while IFS=$'\t' read -r id target arm seed example blen traj; do
    d=$OUT/$target/${arm}_s${seed}
    # A relaunch must not re-run a campaign that already produced trajectories: re-running the same seed writes
    # the same (campaign_seed, trajectory_number) keys again, and a repeated key cannot pair. So a campaign that
    # timed out is left ALONE and its partial trajectories are kept, exactly like the banked 14 -- but it is said
    # out loud, because "rc exists" is not the same as "finished".
    if [ -f "$d/rc" ]; then
      r=$(cat "$d/rc")
      if [ "$r" = 0 ]; then say "  $id already finished, skipping"
      else say "  $id PARTIAL (rc=$r), keeping what it produced and NOT re-running: a repeated seed cannot pair"; fi
      continue
    fi
    mkdir -p "$d"
    say "  launching $id -> $d"
    LAUNCHED=$((LAUNCHED + 1))
    if [ $DRY -eq 1 ]; then echo 0 > "$d/rc"; continue; fi
    # 0.30 is sized for three campaigns sharing one big card. On a 48 GB card at concurrency 1 it hands a single
    # campaign 14 GB, below the ~12 GB per stream the B200 census implies and with no margin, so the fraction has
    # to follow the concurrency rather than be pinned to the three-way case.
    ( XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-0.30} \
      timeout "$CAP" "$PYBIN" "$WT/perf/kitcmp/bc2_gpu.py" \
        --bc2 "$BC2" --out "$d" --seed "$seed" --example "$example" \
        --binder-len "$blen" --max-traj "$traj" --final-designs 1000 \
        > "$d/stdout.log" 2> "$d/stderr.log"; echo $? > "$d/rc" ) &
    PIDS="$PIDS $!"
  done < <(python3 - "$PLAN" "$RREQ" <<'PY'
import json, sys
plan, request = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
by_id = {r["id"]: r for r in request["runs"]}
for rid in plan["waves"][0]["runs"]:
    r = by_id[rid]
    print("\t".join(str(x) for x in (r["id"], r["target"], r["arm"], r["campaign_seed"],
                                     r["example"], r["binder_len"], r["trajectories"])))
PY
  )
  # Every campaign in the wave already had an rc. Without this the re-plan would hand back the same wave
  # for ever, so the session would spin instead of finishing.
  [ "$LAUNCHED" -eq 0 ] && { say "wave $w launched nothing; stopping rather than spinning"; break; }
  for p in $PIDS; do wait "$p"; done

  if [ $DRY -eq 0 ]; then
    JSONLS=$(python3 - "$PLAN" "$RREQ" "$OUT" <<'PY'
import json, os, sys
plan, request, out = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
by_id = {r["id"]: r for r in request["runs"]}
for rid in plan["waves"][0]["runs"]:
    r = by_id[rid]
    print(os.path.join(out, r["target"], f"{r['arm']}_s{r['campaign_seed']}", "runs.jsonl"))
PY
    )
    NWAVE=$(python3 -c "import json;print(len(json.load(open('$PLAN'))['waves'][0]['runs']))")
    M=$(rate_from_runs "$NWAVE" $JSONLS)
    if [ -n "$M" ]; then
      say "wave $w delivered ${M}s per trajectory per stream (was ${BC4_S_PER_TRAJ:-the default}); re-planning on it"
      export BC4_S_PER_TRAJ=$M
    else
      say "wave $w produced no trajectory timings; keeping ${BC4_S_PER_TRAJ:-the default} s/trajectory"
    fi
  fi
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
