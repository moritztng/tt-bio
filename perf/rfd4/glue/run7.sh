#!/bin/bash
# run7: does the one-hot pair gather (+ R, E) hold at 256 and 128? host_split ABAB per size.
#   bash perf/rfd4/glue/run7.sh <chip> <run4 dir>   (reuses run4/main as the base tree, takes the chip lock)
set -u
N=${1:?chip}; R=$(realpath -m "${2:?run4 dir}")
WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
W=$HOME/rfd4-bundle
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> "$R/done7"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp 'lock timeout'; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue
stamp "start chip $N branch $(git -C "$WT" rev-parse --short HEAD)"
for sz in 256 128; do
  for r in 1 2; do
    for arm in main branch; do
      if [ "$arm" = main ]; then t=$R/main; else t=$WT; fi
      (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$WT/perf/rfd4/e2e/host_split.py" \
        --spec "$HOME/rfd4-inputs/pdl1-$sz.yaml" --weights "$W" \
        --out "$R/s$sz-$arm$r" --designs 4 > "$R/s$sz-$arm$r.log" 2>&1)
      stamp "s$sz-$arm$r rc=$?"
    done
  done
done
stamp end
