#!/bin/bash
# The row's after-number: main before pair_build-in-lanes (A) against main with it (B), fair harness, paired ABAB at
# 256 b8/lw4 and b4/lw4, AB at 128 b16, one qb2 chip (not 2, not 3), AICLK sampled by run.py.
#   bash perf/rfd4/p256/after.sh <chip> <outdir> <sha A> <sha B>
set -u
N=$1; R=$2; A=$3; B=$4; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd)
for s in $A $B; do mkdir -p $R/tree-$s && git -C $WT archive $s | tar -x -C $R/tree-$s; done
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3000} --note "RFD4 256 after-number A/B")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N A $A B $B"
fair() {  # tree size batch lane rep
  (cd $R/tree-$1 && PYTHONPATH=$R/tree-$1:$R/tree-$1/perf timeout 1800 $PY perf/rfd4/fair/run.py \
    --spec $IN/pdl1-$2.yaml --weights $W --out $R/f$2-$3-$4-$1-$5 --part p300c --shared-box --max-foreign-cores 99 \
    --quiet-timeout 0 --batch $3 --lane-width $4 --batches 3 > $R/f$2-$3-$4-$1-$5.log 2>&1); stamp "f$2-$3-$4-$1-$5 rc=$?"
}
fair $A 256 8 4 1; fair $B 256 8 4 1; fair $B 256 8 4 2; fair $A 256 8 4 2
fair $A 256 4 4 1; fair $B 256 4 4 1
fair $B 128 16 16 1; fair $A 128 16 16 1
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
