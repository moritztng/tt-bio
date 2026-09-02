#!/bin/sh
# One OpenFold3 deep-MSA rung at a time, on one chip that lsof shows free.
#
# The Galaxy is shared with JapanFold production, so this never takes more than one chip and
# never takes one outside CHIPS -- the set the production pool is not holding. Resume-safe:
# a rung already recorded in ladder.log is skipped, so a relaunch continues the walk.
#
#   sh run_ladder.sh cut_608 cut_640 real_641 tile_704 ...
set -u
WT=/home/cust-team/mthuening/ceilof3
PY=/home/cust-team/mthuening/tt-bio/env/bin/python3.10
OUT=$WT/rundir/out
LOG=$OUT/ladder.log
CHIPS="1 2 3 4 5 6 7 10"
cd "$WT/rundir" || exit 1
mkdir -p "$OUT"

pick() {
  for i in $CHIPS; do
    sudo -n lsof "/dev/tenstorrent/$i" >/dev/null 2>&1 || { echo "$i"; return 0; }
  done
  return 1
}

for r in "$@"; do
  if grep -q "^RUNG $r " "$LOG" 2>/dev/null; then
    echo "=== $r already recorded, skipping" >> "$LOG"
    continue
  fi
  dev=""
  waited=0
  while [ -z "$dev" ]; do
    dev=$(pick) || dev=""
    if [ -z "$dev" ]; then
      waited=$((waited + 1))
      if [ "$waited" -gt 240 ]; then
        echo "RUNG $r rc=- status=nochip wall=- dev=- $(date -u +%FT%TZ)" >> "$LOG"
        exit 3
      fi
      sleep 30
    fi
  done
  s=$(date +%s)
  echo "=== $r start dev=$dev $(date -u +%FT%TZ)" >> "$LOG"
  TT_VISIBLE_DEVICES="$dev" TT_BIO_LEASE_CARDS="$dev" \
    TT_BIO_LEASE_HOLDER=worker:ceiling-openfold3 TT_METAL_LOGGER_LEVEL=FATAL \
    PYTHONPATH="$WT" "$PY" -m tt_bio.main predict "msafix_tile/$r.yaml" \
    --model openfold3 --accelerator tenstorrent --out_dir "$OUT/$r" \
    --msa_dir msacache_deep --msa_cache_only --debug > "$OUT/$r.log" 2>&1
  rc=$?
  e=$(date +%s)
  st=$("$PY" - "$OUT/$r" <<'PYEOF' 2>/dev/null || echo NORESULT
import glob, json, sys
g = glob.glob(sys.argv[1] + "/*/results.json")
print(json.load(open(g[0]))[0]["status"] if g else "NORESULT")
PYEOF
)
  echo "RUNG $r rc=$rc status=$st wall=$((e - s))s dev=$dev $(date -u +%FT%TZ)" >> "$LOG"
done
echo "LADDER DONE $(date -u +%FT%TZ)" >> "$LOG"
