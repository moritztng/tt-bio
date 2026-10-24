#!/bin/bash
# The 224/192 band on the current tree: pass_bench beside the engine call (does pass_bench still take the trimul
# clash path?), then the lane timeline (perf/rfd4/lanes4/split.py) per size: where a design goes between trunk calls.
#   bash perf/rfd4/gap224/split_chain.sh <chip> <outdir> "<n>-w<w>-d<designs> ..."
set -u
N=$1; R=$2; CELLS=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-gap224
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2700} > /dev/null; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 15; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
TT_BIO_TRACE_REGION_MB=512 timeout 900 $PY perf/rfd4/e2e/pass_bench.py --weights $W --n 224 --batch 5 --reps 12 \
  --out $R/pass-224-b5.json > $R/pass-224-b5.log 2>&1; stamp "pass-224-b5 rc=$?"
TT_BIO_TRACE_REGION_MB=2048 timeout 900 $PY perf/rfd4/gap224/call_bench.py --n 224 --batch 5 \
  --out $R/call-224-b5.json > $R/call-224-b5.log 2>&1; stamp "call-224-b5 rc=$?"
for c in $CELLS; do n=${c%%-*}; w=${c#*-w}; w=${w%-d*}; d=${c##*-d}
  timeout 1800 $PY perf/rfd4/lanes4/split.py --spec $IN/pdl1-$n.yaml --weights $W --out $R/split-$c --width $w \
    --designs $d > $R/split-$c.log 2>&1; stamp "split-$c rc=$?"
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
