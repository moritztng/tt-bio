#!/bin/bash
# run5: seeds 17-30 for both arms of run4, so the oracle-free grade has 31 seeds a side.
#   bash perf/rfd4/glue/run5.sh <chip> <run4 dir>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?run4 dir}")
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
SPEC=$HOME/rfd4-inputs/pdl1-512.yaml W=$HOME/rfd4-bundle
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done5"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue
stamp "start chip $N branch $(git -C "$WT" rev-parse --short HEAD)"
for arm in main branch; do
  if [ "$arm" = main ]; then t=$R/main; else t=$WT; fi
  (cd "$t" && PYTHONPATH=$t timeout 5400 "$PY" "$WT/perf/rfd4/e2e/design.py" --spec "$SPEC" --weights "$W" \
    --out "$R/design5/$arm" --rng upstream --seeds $(seq 17 30) > "$R/design5/$arm.log" 2>&1)
  stamp "design5-$arm rc=$?"
done
stamp end
