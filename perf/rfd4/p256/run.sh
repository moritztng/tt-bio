#!/bin/bash
# rfd4-perf-256 census chain on one qb2 chip: device census by family at 256 b1 / b4 / b8, one trunk pass eager vs
# traced at 256 b4 / b8 (how much of a call is dispatch), the host split at 256 w4 / w8, and the sample-width sweep
# around auto_width at 192 / 256 / 320. Books its chip, waits for the window, takes the chip flock.
#   bash perf/rfd4/p256/run.sh <chip> <outdir> [legs]
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-7200} --note "RFD4 256 census + width sweep")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-7200} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
for leg in ${3:-census-b4 pass-b4 split-w4 census-b8 pass-b8 split-w8 census-b1 n192-w4 n192-w7 n192-w8 n256-w4 n256-w8 n320-w1 n320-w2 n320-w4}; do
  case $leg in
    census-b*) b=${leg#census-b}; bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 256 --batch $b \
                && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-256-b$b-v10.json \
                && python3 perf/rfd4/land10/top_ops.py $R/$leg $R/top-ops-256-b$b.json > $R/top-ops-256-b$b.txt ;;
    pass-b*) b=${leg#pass-b}; TT_BIO_TRACE_REGION_MB=2048 timeout 1800 $PY perf/rfd4/e2e/pass_bench.py --weights $W \
                --n 256 --batch $b --reps 12 --out $R/$leg.json ;;
    split-w*) w=${leg#split-w}; timeout 2400 $PY perf/rfd4/lanes4/split.py --spec $IN/pdl1-256.yaml --weights $W \
                --out $R/$leg --width $w --designs 16 ;;
    n*-w*) n=${leg:1:3}; w=${leg#*-w}; timeout 2400 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml \
                --weights $W --out $R/$leg --width $w --designs 16 ;;
  esac > $R/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
