#!/usr/bin/env bash
# Drive a list of OpenBind/OpenFold3 cells on one card. Usage:
#   tt_ob_sweep.sh <card> <tag> <cell> [<cell> ...]
# A cell is "<model>:<input stem>[:samples]", e.g. openbind:ob_apo_512:1
# Skips a cell whose report already exists and carries a device_s_median, so a relaunch resumes.
set -u
CARD="$1"; TAG="$2"; shift 2
WT=/home/ttuser/.coworker/wt/openbind-perf-deep-analysis
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUTDIR=$WT/perf/openbind/tt_results
mkdir -p "$OUTDIR"
cd "$WT" || exit 1
for cell in "$@"; do
  model=${cell%%:*}; rest=${cell#*:}
  stem=${rest%%:*}; samples=${rest##*:}
  [ "$samples" = "$stem" ] && samples=1
  name="${model}_${stem}_s${samples}"
  out=$OUTDIR/$name.json
  if grep -q device_s_median "$out" 2>/dev/null; then echo "SKIP $name (done)"; continue; fi
  echo "=== $name on card $CARD @ $(date -u +%H:%M:%SZ) ==="
  env PYTHONPATH=$WT TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="$CARD" \
      TT_BIO_LEASE_HOLDER=worker:openbind-perf-deep-analysis \
      "$PY" perf/openbind/tt_ob_run.py --model "$model" \
      --input perf/openbind/inputs/$stem.tt.yaml --samples "$samples" \
      --repeat 3 --label "$name" --out "$out" 2>&1 | tail -40
  echo "=== $name rc=$? @ $(date -u +%H:%M:%SZ) ==="
done
echo "SWEEP $TAG COMPLETE @ $(date -u +%H:%M:%SZ)"
