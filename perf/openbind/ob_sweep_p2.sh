#!/usr/bin/env bash
# Drive OpenBind/OpenFold3 cells on one card from THIS worktree. Usage:
#   ob_sweep_p2.sh <card> <tag> <cell> [<cell> ...]
# A cell is "<model>:<input stem>[:samples]". Skips a cell whose report already carries a
# device_s_median, so a relaunch resumes. perf/openbind/tt_ob_sweep.sh hardcodes the pass-1
# worktree path, which would silently fold a different tree.
set -u
CARD="$1"; TAG="$2"; shift 2
REPEAT=${REPEAT:-3}   # warm folds after the discarded cold one
WT=/home/ttuser/.coworker/wt/openbind-perf-p2
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
  env PYTHONPATH=$WT TT_VISIBLE_DEVICES="$CARD" TT_BIO_LEASE_CARDS="0,$CARD" \
      TT_BIO_LEASE_HOLDER=worker:openbind-perf-p2 \
      "$PY" perf/openbind/tt_ob_run.py --model "$model" \
      --input perf/openbind/inputs/$stem.tt.yaml --samples "$samples" \
      --repeat "$REPEAT" --label "$name" --out "$out" 2>&1 | tail -30
  echo "=== $name rc=$? @ $(date -u +%H:%M:%SZ) ==="
done
echo "SWEEP $TAG COMPLETE @ $(date -u +%H:%M:%SZ)"
