#!/bin/bash
# Bet PB on the card: pair_build at lane width > 1. Runs from frozen snapshots (git archive) of A (before) and B
# (after), so an edit to the worktree cannot reach a running leg. Device test, synced attribution, then the fair
# harness ABAB at 256 b8/lw4 and AB at 128 b16/lw16.
#   bash perf/rfd4/p256/pb.sh <chip> <outdir> <sha A> <sha B>
set -u
N=$1; R=$2; A=$3; B=$4; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R; R=$(cd $R && pwd)
for s in $A $B; do mkdir -p $R/tree-$s && git -C $WT archive $s | tar -x -C $R/tree-$s; done
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-2700} --note "RFD4 256 pair_build lanes A/B")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2700} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N A $A B $B"
intree() { cd $R/tree-$1 && PYTHONPATH=$R/tree-$1:$R/tree-$1/perf "${@:2}"; }
intree $B timeout 900 $PY -m pytest -q -m device tests/test_rfd4_pair_device.py -k one_program > $R/test-B.log 2>&1
stamp "test-B rc=$?"
intree $B timeout 900 $PY perf/rfd4/p256/callhead.py --spec $IN/pdl1-256.yaml --weights $W --out $R/head-B --width 4 \
  --reps 2 > $R/head-B.log 2>&1; stamp "head-B rc=$?"
fair() {  # tree size batch lane
  intree $1 timeout 1800 $PY perf/rfd4/fair/run.py --spec $IN/pdl1-$2.yaml --weights $W --out $R/f$2-$3-$4-$1-$5 \
    --part p300c --shared-box --max-foreign-cores 99 --quiet-timeout 0 --batch $3 --lane-width $4 --batches 3 \
    > $R/f$2-$3-$4-$1-$5.log 2>&1; stamp "f$2-$3-$4-$1-$5 rc=$?"
}
fair $A 256 8 4 1; fair $B 256 8 4 1; fair $A 256 8 4 2; fair $B 256 8 4 2
fair $A 128 16 16 1; fair $B 128 16 16 1
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
