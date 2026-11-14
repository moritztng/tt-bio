#!/bin/bash
# After the uneven-lanes fix: the leg that crashed (192, width 7, 16 designs = lanes 6, 6, 4) on the device, then 256
# b4/lw4 and b8/lw4 interleaved twice on a quiet chip (r2 read them under 11 foreign cores).
#   bash perf/rfd4/p256/rerun.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3000} --note "RFD4 256 lanes fix smoke + clean L A/B")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
timeout 2400 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-192.yaml --weights $W --out $R/n192-w7 --width 7 \
  --designs 16 > $R/n192-w7.log 2>&1; stamp "n192-w7 rc=$?"
for arm in 256:4:4 256:8:4 256:4:4b 256:8:4b; do
  IFS=: read n b lw <<< "$arm"; lw=${lw%b}
  timeout 1800 $PY perf/rfd4/fair/run.py --spec $IN/pdl1-$n.yaml --weights $W --out $R/${arm//:/-} --part p300c \
    --shared-box --max-foreign-cores 99 --quiet-timeout 0 --batch $b --lane-width $lw --batches 3 > $R/${arm//:/-}.log 2>&1
  stamp "$arm rc=$?"
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
