#!/bin/bash
# rfd4-perf-tri224 run2 on one qb2 chip (perf only, no grade): pair_x8's move back exact on the device, then the paired
# traced trunk, main's default vs + pair_x8, at main's widths (224 b10, 192 b14, 256 b4) and the batch-1 guards 384 / 512.
#   bash perf/rfd4/tri224/run2.sh <chip> <outdir> [rounds]
set -u
N=$1; R=$2; ROUNDS=${3:-2}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=${OWN:-rfd4-perf-tri224}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
export TT_BIO_TRACE_REGION_MB=${TT_BIO_TRACE_REGION_MB:-2048}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3600} --note "tri224 run2" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(basename $WT)"
timeout 900 $PY perf/rfd4/tri224/x8_exact.py --out $R/x8_exact.json > $R/x8_exact.log 2>&1
stamp "x8_exact rc=$?"
for r in $(seq 1 $ROUNDS); do
  for leg in 224x10 192x14 256x4 384x1 512x1; do
    n=${leg%x*}; b=${leg#*x}
    timeout 1800 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --n $n --batch $b --reps ${REPS:-30} --levers ${LEVERS:-+pair_x8} \
      --out $R/ab$n-b$b-r$r.json > $R/ab$n-b$b-r$r.log 2>&1
    stamp "ab$n-b$b-r$r rc=$? $(python3 -c "import json;d=json.load(open('$R/ab$n-b$b-r$r.json'));print(d['paired_speedup_median'],d['aiclk'])" 2>/dev/null)"
  done
done
exec 9>&-; $BOOK rm $OWN > /dev/null
stamp ALLDONE
