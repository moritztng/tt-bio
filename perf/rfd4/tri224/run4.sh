#!/bin/bash
# rfd4-perf-tri224 run4 on one qb2 chip (perf only, no grade): lever Y (`pair_x8t`, the move writes bfp8 and the tail reads
# it) on top of X: the move's bfp8-out error and time, then the paired traced trunk default + pair_x8 vs + pair_x8,pair_x8t.
#   OWN=... bash perf/rfd4/tri224/run4.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=${OWN:-rfd4-perf-tri224-ab}
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri224 PYTHONPATH=$WT:$WT/perf
export TT_BIO_TRACE_REGION_MB=${TT_BIO_TRACE_REGION_MB:-2048}
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-1800} --note "tri224 run4 Y" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(basename $WT)"
timeout 900 $PY perf/rfd4/tri224/x8_exact.py --out $R/x8_exact.json > $R/x8_exact.log 2>&1
stamp "x8_exact rc=$?"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as L; print(','.join(sorted(L | {'pair_x8'})))")
for leg in 224x10 192x14 256x4; do
  n=${leg%x*}; b=${leg#*x}
  timeout 1800 $PY perf/rfd4/bfp8/stack_ab.py --weights $W --n $n --batch $b --reps ${REPS:-30} --base $B --levers +pair_x8t \
    --out $R/ab$n-b$b.json > $R/ab$n-b$b.log 2>&1
  stamp "ab$n-b$b rc=$? $(python3 -c "import json;d=json.load(open('$R/ab$n-b$b.json'));print(d['paired_speedup_median'],d['z_rel_rms_stack_vs_base'],d['aiclk'])" 2>/dev/null)"
done
exec 9>&-; $BOOK rm $OWN > /dev/null
stamp ALLDONE
