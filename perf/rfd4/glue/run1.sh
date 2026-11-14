#!/bin/bash
# run1: whole-call census at 512 (call_prof) then the design split at 512 (host_split, 1 compile + 3 warm) on this tree.
#   bash perf/rfd4/glue/run1.sh <chip> <outdir>     (takes ~/spd_qb2_chip<chip>.lock itself)
set -u
N=${1:?chip}; R=${2:?outdir}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
mkdir -p "$R"; cd "$WT"
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { echo "lock timeout $(date -u +%FT%TZ)" >> "$R/done"; exit 1; }
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
stamp "start chip $N sha $(git rev-parse --short HEAD)"
bash perf/rfd4/glue/call_prof.sh "$R/call-512" "$N" --n 512 > "$R/call-512.log" 2>&1; stamp "call512 rc=$?"
TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue PYTHONPATH=$WT \
  timeout 2400 $PY perf/rfd4/e2e/host_split.py --spec $HOME/rfd4-inputs/pdl1-512.yaml --weights $HOME/rfd4-bundle \
  --out "$R/split-512" --designs 4 > "$R/split-512.log" 2>&1; stamp "split512 rc=$?"
stamp end
