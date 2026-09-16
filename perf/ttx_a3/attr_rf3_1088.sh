#!/usr/bin/env bash
# The decisive arm: rf3 at 1088 tokens, where the lever actually fires.
#
# Every other cell in this release gate either sits at or below the 1024-token cap, where the route
# is unreachable by construction, or is a capacity cell that only asks whether the fold allocates.
# rf3/1088 is the one shape the gate folds where TT_BIO_SDPA_FUSED_LARGE_S changes which kernel
# runs, and the census proves it rather than assuming it:
#
#   census_rf3-1024-rep0    SDPA_FUSED_LARGE_S  served 0     declined 0    <- unreachable
#   census_rf3-1088-warmup  SDPA_FUSED_LARGE_S  served 1088  declined 0    <- every call
#
# That warmup fold completed rc=0 in 215 s. The NEXT fold at the same shape with the same flag hung
# past 1000 s at 102 % CPU on a box at loadavg 1.00. Same shape, same flag, one completes and one
# does not, so the question is a rate, not a yes/no, and a single pair of arms cannot answer it.
#
# So: alternating arms, separate processes, several reps each. Interleaved rather than blocked, so
# a drift in the box over the hour cannot be read as an arm difference. HUNG is a classification,
# not a crash: 600 s is about 3x the healthy wall, and a fold past it is recorded as HUNG and killed
# so the next rep gets the card.
#
# Reuses scripts/lever_census.py -- the gate's own instrument -- so every rep also records the
# served/declined counts. That makes the off arm a real negative control: it must read served 0.
set -u
WT=/home/ttuser/.coworker/wt/ttx-a3-sdpa-ship-remerge
cd "$WT" || exit 1
OUT="$WT/perf/ttx_a3/gate2/rf3_1088"
PROG="$OUT/progress"
CARD=0
P=/home/ttuser/tt-bio-dev/env/bin/python3
CENSUS="$WT/scripts/lever_census.py"
FIXTURE="$WT/perf/size512/fixtures/cdk2x2_1088.yaml"
mkdir -p "$OUT"; touch "$PROG"

export PYTHONPATH="$WT"
export TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-a3-sdpa-ship-remerge

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$PROG"; }

rep() {  # $1 = name, $2 = on|off
  local name="$1" arm="$2"
  grep -q " $name rc=" "$PROG" && { echo "skip $name"; return 0; }
  local ev=()
  [ "$arm" = off ] && ev=(TT_BIO_SDPA_FUSED_LARGE_S=0)
  log "$name START arm=$arm loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  local t0 t1
  t0=$(date +%s)
  env "${ev[@]}" TT_VISIBLE_DEVICES=$CARD timeout -k 30 600 \
    "$P" "$CENSUS" --tt-bio "$P" --label "rf3-1088-$name" \
    --out "$OUT/census_$name.json" -- \
    -m tt_bio.main predict "$FIXTURE" --model rf3 --single_sequence \
    --sampling_steps 6 --diffusion_samples 1 --seed 0 \
    --out_dir "$OUT/out_$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  t1=$(date +%s)
  local served
  served=$($P - <<PY 2>/dev/null || echo "?"
import json
try:
    d = json.load(open("$OUT/census_$name.json"))
    r = [x for x in d["rows"] if x["flag"] == "SDPA_FUSED_LARGE_S"][0]
    print("served=%s declined=%s resolved=%s" % (r["served"], r["declined"], r["resolved"]))
except Exception as e:
    print("census-unreadable")
PY
)
  [ "$rc" = 124 ] && log "$name rc=124 HUNG wall=$((t1-t0))s arm=$arm $served" \
                  || log "$name rc=$rc wall=$((t1-t0))s arm=$arm $served"
}

rep off1 off
rep on1  on
rep off2 off
rep on2  on
rep off3 off
rep on3  on
rep off4 off
rep on4  on
log "RF3_1088_DONE"
