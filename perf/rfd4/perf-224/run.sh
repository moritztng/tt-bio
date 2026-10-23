#!/bin/bash
# rfd4-perf-224 chain on one qb1 p150a: census v10 by op family at 192 / 224 (device profiler, Tracy build of the
# wheel ttnn in ~/tt-metal-b2z), one trunk call eager vs traced, and designs at widths around auto_width.
# Waits for the book.py window on its card, takes the card flock and an idle device node.
#   bash perf/rfd4/perf-224/run.sh <card> <outdir> "<legs>"
#   legs: census-<n>-b<w> | pass-<n>-b<w> | des-<n>-w<w>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-fc773b53a8ac/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-224
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN 0,1 ${HOLD_S:-10800} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb1_card$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start card $N tree $(git rev-parse --short HEAD)"
for leg in $3; do
  n=$(echo $leg | cut -d- -f2); w=${leg##*[bw]}
  case $leg in
    census-*) for i in $(seq 120); do grep -q BUILD-DONE $HOME/scratch/rfd4-224/build_b2z.log && break; sleep 60; done
              bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n $n --batch $w \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-$n-b$w-v10.json \
                && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-$n-b$w.json > $R/top-ops-$n-b$w.txt ;;
    pass-*) TT_BIO_TRACE_REGION_MB=2048 timeout 1800 $PY perf/rfd4/e2e/pass_bench.py --weights $W \
                --n $n --batch $w --reps 12 --out $R/$leg.json ;;
    des-*) timeout 2400 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --width $w --designs ${DESIGNS:-20} ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; stamp ALLDONE
