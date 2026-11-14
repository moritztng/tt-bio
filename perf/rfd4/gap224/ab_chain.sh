#!/bin/bash
# Lever P on the card: write_ab.py per size, one process per size, on one booked qb2 chip.
#   bash perf/rfd4/gap224/ab_chain.sh <chip> <outdir> "<n>-b<batch>-r<rounds> ..."
set -u
N=$1; R=$2; CELLS=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=${PY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-gap224
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3600} > /dev/null; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 15; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
for c in $CELLS; do n=${c%%-*}; b=${c#*-b}; b=${b%-r*}; r=${c##*-r}
  TT_BIO_TRACE_REGION_MB=2048 timeout 2400 $PY perf/rfd4/gap224/write_ab.py --spec $IN/pdl1-$n.yaml --weights $W \
    --batch $b --rounds $r --out $R/ab-$c > $R/ab-$c.log 2>&1; stamp "ab-$c rc=$?"
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
