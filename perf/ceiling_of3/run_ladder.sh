#!/bin/sh
# One OpenFold3 deep-MSA rung at a time, on one chip that lsof shows free.
#
# The Galaxy is shared with JapanFold production, so this never takes more than one chip and
# never takes one outside CHIPS -- the set the production pool is not holding.
#
# lsof and the device open are not one atomic step, so a sibling can take the chip in between.
# tt-bio's own lease refuses that open (exit 75, DeviceInUseError) and NOTHING runs. That is a
# non-result, not a rung: it is logged as UNRUN and retried on another chip, because the resume
# check below matches `^RUNG ` only. Recording a lost race as a rung would make a relaunch skip
# a size that was never folded, which is how a ceiling gets published with a hole in its ladder.
#
#   sh run_ladder.sh cut_608 cut_640 real_641 tile_704 ...
set -u
WT=/home/cust-team/mthuening/ceilof3
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
OUT=$WT/rundir/out
LOG=$OUT/ladder.log
# Descending. Sibling ceiling tasks scan ascending, and picking the same first-free chip
# lost the device race three times on chip 6 at ~2 min a loss (the lease waits 120 s before
# refusing). Scanning from the other end costs nothing and stops the collision.
CHIPS="10 7 6 5 4 3 2 1"
MAX_CONTENTION=12
cd "$WT/rundir" || exit 1
mkdir -p "$OUT"

pick() {
  for i in $CHIPS; do
    sudo -n lsof "/dev/tenstorrent/$i" >/dev/null 2>&1 || { echo "$i"; return 0; }
  done
  return 1
}

wait_for_chip() {
  waited=0
  while : ; do
    dev=$(pick) && { echo "$dev"; return 0; }
    waited=$((waited + 1))
    [ "$waited" -gt 240 ] && return 1
    sleep 30
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
    if [ "$attempt" -gt "$MAX_CONTENTION" ]; then
      echo "UNRUN $r contention x$MAX_CONTENTION $(date -u +%FT%TZ)" >> "$LOG"
      break
    fi
    dev=$(wait_for_chip) || {
      echo "UNRUN $r no-free-chip-in-2h $(date -u +%FT%TZ)" >> "$LOG"
      exit 3
    }
    s=$(date +%s)
    echo "=== $r attempt $attempt start dev=$dev $(date -u +%FT%TZ)" >> "$LOG"
    TT_VISIBLE_DEVICES="$dev" TT_BIO_LEASE_CARDS="$dev" \
      TT_BIO_LEASE_HOLDER=worker:ceiling-openfold3 TT_METAL_LOGGER_LEVEL=FATAL \
      PYTHONPATH="$WT" "$PY" -m tt_bio.main predict "msafix_tile/$r.yaml" \
      --model openfold3 --accelerator tenstorrent --out_dir "$OUT/$r" \
      --msa_dir msacache_deep --msa_cache_only --debug > "$OUT/$r.log" 2>&1
    rc=$?
    e=$(date +%s)
    if grep -q "DeviceInUseError" "$OUT/$r.log" 2>/dev/null; then
      echo "UNRUN $r attempt $attempt lost dev=$dev to a lease holder $(date -u +%FT%TZ)" >> "$LOG"
      sleep 20
      continue
    fi
    st=$("$PY" - "$OUT/$r" <<'PYEOF' 2>/dev/null || echo NORESULT
import glob, json, sys
g = glob.glob(sys.argv[1] + "/*/results.json")
print(json.load(open(g[0]))[0]["status"] if g else "NORESULT")
PYEOF
)
    echo "RUNG $r rc=$rc status=$st wall=$((e - s))s dev=$dev $(date -u +%FT%TZ)" >> "$LOG"
    break
  done
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
