#!/bin/bash
# Bet PB, second chain, off qb2 chip 3 (open determinism marker): the 32-design seed-matched grade at 256 w4,
# the 128 b16 fair ABAB that r7 lost to the 20:57Z host reboot, and 512 width 1 byte identity (PB leaves width 1
# alone, so A and B must write the same CIF bytes). Card work first, CPU grading after the booking is released.
#   bash perf/rfd4/p256/pb2.sh <chip> <outdir> <sha A> <sha B>
set -u
N=$1; R=$2; A=$3; B=$4; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-256
mkdir -p $R/grade; R=$(cd $R && pwd)
for s in $A $B; do mkdir -p $R/tree-$s && git -C $WT archive $s | tar -x -C $R/tree-$s; done
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3000} --note "RFD4 256 pair_build grade + 128 A/B + 512 identity")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N A $A B $B"
intree() { cd $R/tree-$1 && PYTHONPATH=$R/tree-$1:$R/tree-$1/perf "${@:2}"; }
designs() {  # tree size width count
  intree $1 timeout 1800 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$2.yaml --weights $W --out $R/d$2-$1 \
    --width $3 --designs $4 --seed 0 > $R/d$2-$1.log 2>&1; stamp "d$2-$1 rc=$?"
}
fair() {  # tree size batch lane
  intree $1 timeout 1800 $PY perf/rfd4/fair/run.py --spec $IN/pdl1-$2.yaml --weights $W --out $R/f$2-$3-$4-$1-$5 \
    --part p300c --shared-box --max-foreign-cores 99 --quiet-timeout 0 --batch $3 --lane-width $4 --batches 3 \
    > $R/f$2-$3-$4-$1-$5.log 2>&1; stamp "f$2-$3-$4-$1-$5 rc=$?"
}
designs $A 256 4 32; designs $B 256 4 32
fair $A 128 16 16 1; fair $B 128 16 16 1; fair $A 128 16 16 2; fair $B 128 16 16 2
designs $A 512 1 2; designs $B 512 1 2
exec 9>&-; $BOOK rm $OWN; stamp released
# _entry.time is the write clock, the only line two identical designs may differ in
nt() { grep -v '^_entry.time' $R/d512-$1/pdl1_binder_512_d$2.cif; }
for k in 0 1; do cmp -s <(nt $A $k) <(nt $B $k); stamp "512 d$k cmp=$?"; done
cd $R/tree-$B
f() { echo $R/d256-$1/pdl1_binder_256_d$2.cif; }
for k in $(seq 0 31); do
  $PY perf/rfd4/e2e/grade.py --ours $(f $B $k) --upstream $(f $A $k) --floor $(f $A $(( (k+1)%32 ))) \
    --out $R/grade/d$k.json > /dev/null 2>&1
done
cp $HOME/scratch/rfd4-land5/QUALITY.base.json $R/quality256.json
$PY perf/rfd4/quality.py grade --feature binder --tt $R/d256-$B --ref $R/d256-$A --allow-no-oracle \
  --label "RFD4 pair_build in lanes (B $B) vs fallback (A $A), 256 PD-L1 binder, 32 designs w4, qb2 chip $N" \
  --out $R/quality256.json > $R/quality256.log 2>&1
stamp "grade rc=$?"; stamp ALLDONE
