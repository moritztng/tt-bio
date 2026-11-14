#!/bin/bash
# rfd4-perf-tri224 run1 on one qb2 chip (perf only, no grade): price.py (move back stage ablations, contraction
# output bf16 vs bfp8) at main's widths, then census v11 by op family at 224 b10 and 192 b14 (Tracy build, elt_prof.sh).
#   bash perf/rfd4/tri224/run1.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-tri224
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2400} --note "tri224 run1" > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(basename $WT)"
timeout 900 $PY perf/rfd4/tri224/price.py --out $R/price.json > $R/price.log 2>&1
stamp "price rc=$?"
for leg in census-224-b10 census-192-b14; do
  n=$(echo $leg | cut -d- -f2); w=${leg##*b}
  bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n --batch $w > $R/$leg.log 2>&1 \
    && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-$n-b$w-v11.json >> $R/$leg.log 2>&1 \
    && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-$n-b$w.json > $R/top-ops-$n-b$w.txt 2>> $R/$leg.log
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN > /dev/null
stamp ALLDONE
