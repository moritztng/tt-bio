#!/bin/sh
# One OpenFold3 deep-MSA rung at a time, on one chip that is free on all three counts
# pick_chip.py checks. The Galaxy is shared with JapanFold production, so this never takes
# more than one chip and never takes one inside the pool's own grant.
#
#   TREE=/path/to/checkout OUT=rundir/out_fix sh run_ladder.sh tile_768 tile_800 ...
#
# TREE is the code under test and OUT is that arm's own log dir, so the baseline and the
# fix arm share one fixture set (rundir/msafix_tile, rundir/msacache_deep) and cannot mix
# results. Rungs are resume-safe: a relaunch skips a size only when a `RUNG ` line exists
# for it, so a lost device race -- which runs nothing at all -- can never be mistaken for a
# measurement and leave a hole in the ladder.
set -u
RUN=${RUN:-/home/cust-team/mthuening/ceilof3/rundir}
TREE=${TREE:-/home/cust-team/mthuening/ceilof3}
OUT=${OUT:-$RUN/out}
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
LOG=$OUT/ladder.log
# A lost race costs this much wall clock and produces nothing, so keep it short. The 120 s
# default is sized for waiting out a real co-tenant; here there is always another chip.
LEASE_TIMEOUT=${LEASE_TIMEOUT:-20}
WAIT_LOOPS=${WAIT_LOOPS:-240}
cd "$RUN" || exit 1
mkdir -p "$OUT"

wait_for_chip() {
  waited=0
  while : ; do
    dev=$("$PY" "$TREE/perf/ceiling_of3/pick_chip.py" 2>/dev/null) && { echo "$dev"; return 0; }
    waited=$((waited + 1))
    [ "$waited" -gt "$WAIT_LOOPS" ] && return 1
    sleep 15
  done
}

for r in "$@"; do
  if grep -q "^RUNG $r " "$LOG" 2>/dev/null; then
    echo "=== $r already folded, skipping" >> "$LOG"
    continue
  fi
  attempt=0
  while : ; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 12 ]; then
      echo "UNRUN $r contention x12 $(date -u +%FT%TZ)" >> "$LOG"
      break
    fi
    dev=$(wait_for_chip) || {
      echo "UNRUN $r no-free-chip $(date -u +%FT%TZ)" >> "$LOG"
      exit 3
    }
    node=$("$PY" "$TREE/perf/ceiling_of3/pick_chip.py" --map | tr ' ' '\n' | grep "^$dev->" )
    s=$(date +%s)
    echo "=== $r attempt $attempt start card=$dev ($node) tree=$(git -C "$TREE" rev-parse --short HEAD) $(date -u +%FT%TZ)" >> "$LOG"
    TT_VISIBLE_DEVICES="$dev" TT_BIO_LEASE_CARDS="$dev" \
      TT_BIO_LEASE_HOLDER=worker:ceiling-openfold3 TT_BIO_LEASE_TIMEOUT="$LEASE_TIMEOUT" \
      TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" \
      "$PY" -m tt_bio.main predict "msafix_tile/$r.yaml" \
      --model openfold3 --accelerator tenstorrent --out_dir "$OUT/$r" \
      --msa_dir msacache_deep --msa_cache_only --debug > "$OUT/$r.log" 2>&1
    rc=$?
    e=$(date +%s)
    if grep -q "DeviceInUseError" "$OUT/$r.log" 2>/dev/null; then
      echo "UNRUN $r attempt $attempt lost card=$dev to a lease holder $(date -u +%FT%TZ)" >> "$LOG"
      continue
    fi
    st=$("$PY" - "$OUT/$r" <<'PYEOF' 2>/dev/null || echo NORESULT
import glob, json, sys
g = glob.glob(sys.argv[1] + "/*/results.json")
print(json.load(open(g[0]))[0]["status"] if g else "NORESULT")
PYEOF
)
    echo "RUNG $r rc=$rc status=$st wall=$((e - s))s card=$dev $(date -u +%FT%TZ)" >> "$LOG"
    break
  done
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
