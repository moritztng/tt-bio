#!/bin/bash
# Bet PB quality: 32 seed-matched designs at 256 in 4-wide lanes (rng per_design, seed 0) on A and B, from frozen
# snapshots; then grade.py per design (B vs A at the same seed, floor = A at the next) and quality.py binder pooled
# against land5's pdl1-256 seed floor. Width 1 is untouched by PB, so the grade has to run in lanes.
#   bash perf/rfd4/p256/grade.sh <chip> <outdir> <sha A> <sha B>
set -u
N=$1; R=$2; A=$3; B=$4; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R/grade; R=$(cd $R && pwd)
for s in $A $B; do mkdir -p $R/tree-$s && git -C $WT archive $s | tar -x -C $R/tree-$s; done
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-1500} --note "RFD4 256 pair_build lanes quality")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-1500} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N A $A B $B"
for s in $A $B; do
  (cd $R/tree-$s && PYTHONPATH=$R/tree-$s:$R/tree-$s/perf timeout 1800 $PY perf/rfd4/b128/designs.py \
    --spec $IN/pdl1-256.yaml --weights $W --out $R/d-$s --width 4 --designs 32 --seed 0 > $R/d-$s.log 2>&1)
  stamp "designs $s rc=$?"
done
exec 9>&-; $BOOK rm $OWN
cd $R/tree-$B
f() { echo $R/d-$1/pdl1_binder_256_d$2.cif; }
for k in $(seq 0 31); do
  $PY perf/rfd4/e2e/grade.py --ours $(f $B $k) --upstream $(f $A $k) --floor $(f $A $(( (k+1)%32 ))) \
    --out $R/grade/d$k.json > /dev/null 2>&1
done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality256.json
$PY perf/rfd4/quality.py grade --feature binder --tt $R/d-$B --ref $R/d-$A --allow-no-oracle \
  --label "RFD4 pair_build in lanes (B $B) vs fallback (A $A), 256 PD-L1 binder, 32 designs w4" \
  --out $R/quality256.json > $R/quality256.log 2>&1
stamp "grade rc=$?"; stamp ALLDONE
