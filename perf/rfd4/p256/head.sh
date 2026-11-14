#!/bin/bash
# Attribute the ~35 ms a lane call costs beyond the trunk kernels at 256 (r1 timeline): callhead.py (every piece
# synced) at 256 w4, then the unsynced lane split at w4 (16 designs) on the same tree and clock for the excess.
#   bash perf/rfd4/p256/head.sh <chip> <outdir>
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$WT:$WT/perf
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-1500} --note "RFD4 256 call-head attribution")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-1500} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
timeout 1200 $PY perf/rfd4/p256/callhead.py --spec $IN/pdl1-256.yaml --weights $W --out $R/head-256-w4 --width 4 \
  --reps 2 > $R/head-256-w4.log 2>&1; stamp "head-256-w4 rc=$?"
timeout 1800 $PY perf/rfd4/lanes4/split.py --spec $IN/pdl1-256.yaml --weights $W --out $R/split-w4 --width 4 \
  --designs 16 > $R/split-w4.log 2>&1; stamp "split-w4 rc=$?"
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
