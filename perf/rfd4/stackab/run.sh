#!/bin/bash
# stackab: whole fair-config designs, fair7's tree (A) against main (B), on one qb2 chip.
# Per size: one warm leg per arm (--batches 1, not read), then legs A B B A A B, 4 warm batches each, same seeds
# every leg. Each tree runs its own perf/rfd4/fair/run.py, as fair7 did. The 512 A tree (95fe840d0) gets the one
# harness line 0a3ee1291 added (--shared-box skips the tt-smi board probe, which walks every chip on a shared box);
# nothing under tt_bio/ is touched.
#   bash perf/rfd4/stackab/run.sh <chip> <outdir> <main sha>      (detach it)
set -u
N=$1; R=$2; BSHA=$3; WT=$(cd "$(dirname "$0")/../../.." && pwd)
PY=$HOME/tt-bio-dev/env/bin/python; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-stackab
S=$HOME/scratch/rfd4-stackab/trees
mkdir -p $R $S; R=$(cd $R && pwd)
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d' ' -f1 /proc/loadavg)" >> $R/done; }
tree() {  # tree <name> <sha>
  [ -d $S/$1 ] || git -C $WT worktree add -q --detach $S/$1 $2 || exit 4
  [ "$(git -C $S/$1 rev-parse HEAD)" = "$(git -C $WT rev-parse $2)" ] || exit 4
}
tree A512 95fe840d0; tree A224 0a3ee1291; tree B $BSHA
git -C $WT show 0a3ee1291:perf/rfd4/fair/run.py > $S/A512/perf/rfd4/fair/run.py
[ "$(git -C $S/A512 diff --stat -- tt_bio | wc -l)" = 0 ] || exit 4
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-18000} --note "stackab: fair7 tree vs main, 512+224 whole designs")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-18000} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N A512 95fe840d0 A224 0a3ee1291 B $BSHA"
( while sleep 1; do echo "$(date +%s) $(cat "/sys/class/tenstorrent/tenstorrent!$N/tt_aiclk")"; done ) > $R/aiclk.txt &
CLK=$!
leg() {  # leg <size> <arm> <name> <batches>
  local T=$S/B; [ $2 = A ] && T=$S/A$1
  local W="--batch 4 --lane-width 1"; [ $1 = 224 ] && W="--batch 20 --lane-width 10"
  stamp "$3 begin"
  (cd $T && PYTHONPATH=$T timeout 3600 $PY -c "import tt_bio; print('tt_bio', tt_bio.__file__)" > $R/$3.log 2>&1
   PYTHONPATH=$T timeout 3600 $PY perf/rfd4/fair/run.py --spec $HOME/rfd4-inputs/pdl1-$1.yaml \
    --weights $HOME/rfd4-bundle --out $R/raw/$3 --part p300c --shared-box --quiet-timeout ${QT:-600} \
    $W --batches $4 --max-attempts 4 >> $R/$3.log 2>&1)
  stamp "$3 rc=$?"
  $BOOK add $OWN $N ${HOLD_S:-18000} > /dev/null
}
for s in ${SIZES:-512 224}; do
  leg $s A warm${s}A 1; leg $s B warm${s}B 1
  k=0; for arm in A B B A A B; do k=$((k+1)); leg $s $arm s${s}_$k$arm 4; done
done
kill $CLK
exec 9>&-; $BOOK rm $OWN
for t in A512 A224 B; do git -C $WT worktree remove --force $S/$t; done
stamp ALLDONE
