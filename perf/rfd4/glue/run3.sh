#!/bin/bash
# run3: pair-build variants at 512 and 256 (build_bench.py), under the chip flock.
#   bash perf/rfd4/glue/run3.sh <chip> <outdir>
set -u
N=${1:?chip}; R=$(realpath -m "${2:?outdir}"); WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
mkdir -p "$R"; cd "$WT"
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> "$R/done"; }
exec 9>"$HOME/spd_qb2_chip$N.lock"
flock -w 10800 9 || { stamp "lock timeout"; exit 1; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-glue PYTHONPATH=$WT
stamp "start chip $N $(git rev-parse --short HEAD)"
for n in 512 256; do
  timeout 1200 $PY perf/rfd4/glue/build_bench.py --n $n --out "$R/build-$n.json" > "$R/build-$n.log" 2>&1
  stamp "build$n rc=$?"
done
stamp end
