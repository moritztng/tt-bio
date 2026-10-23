#!/bin/bash
# run3: repeat run2's 512 ABAB (its round 2 drooped to 1131-1231 MHz on both arms) from run2's snapshots.
#   bash perf/rfd4/host2/run3.sh <chip> <run2 dir> <outdir>
set -u
N=${1:?chip}; S=$(realpath "${2:?run2dir}"); R=$(realpath -m "${3:?outdir}")
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
W=$HOME/rfd4-bundle; mkdir -p "$R"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 3600 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-host2
stamp "start chip $N"
for r in 3 4; do
  for arm in main branch; do
    t=$S/$arm
    (cd "$t" && PYTHONPATH=$t timeout 2400 "$PY" "$t/perf/rfd4/e2e/host_split.py" \
      --spec "$HOME/rfd4-inputs/pdl1-512.yaml" --weights "$W" \
      --out "$R/s512-$arm$r" --designs 4 > "$R/s512-$arm$r.log" 2>&1)
    stamp "s512-$arm$r rc=$?"
  done
done
stamp end
