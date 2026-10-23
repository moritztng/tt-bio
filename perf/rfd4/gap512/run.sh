#!/bin/bash
# rfd4-perf-gap512 split chain on one qb2 chip: where a 512 design run as width-1 lanes spends wall that is not device
# time. Arms: A = this tree (main), L = A + lanes4 lever (heads after the next launch, lanes4-lever.patch).
# Legs: split.py at 512 width 1 (--designs lanes in flight), ABAB, then census512 (device time, same tree as A).
#   bash run.sh <chip> <outdir> [legs]        legs default: "sA1 sL1 sA2 sL2 census512"
set -u
N=$1; R=$2; WT=$(cd "$(dirname "$0")/../../.." && pwd); OWN=rfd4-perf-gap512
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle; BOOK="python3 $HOME/spd/feed/book.py"; D=${DESIGNS:-4}; SZ=${SIZE:-512}
mkdir -p $R; R=$(cd $R && pwd)
for arm in A L; do t=$R/tree-$arm; [ -d $t ] && continue; mkdir -p $t; git -C $WT archive HEAD | tar -x -C $t
  git -C $WT rev-parse HEAD > $t/TREE; done
[ -f $R/tree-L/.patched ] || { (cd $R/tree-L && git apply perf/rfd4/gap512/lanes4-lever.patch) && touch $R/tree-L/.patched; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
stamp "booked $($BOOK add $OWN $N ${HOLD_S:-3600} --note "RFD4 gap512 split")"
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-3600} > /dev/null; sleep 60; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 7200 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 20; done
stamp "start chip $N tree $(cat $R/tree-A/TREE)"
for leg in ${3:-sA1 sL1 sA2 sL2 census512}; do
  case $leg in
    s[AL]*) arm=${leg:1:1}; t=$R/tree-$arm
      (cd $t && PYTHONPATH=$t:$t/perf timeout 1800 $PY perf/rfd4/gap512/split.py --spec $IN/pdl1-$SZ.yaml --weights $W \
         --out $R/$leg --width 1 --designs $D) > $R/$leg.log 2>&1 ;;
    census512) (cd $R/tree-A && bash perf/rfd4/e2e/elt_prof.sh $R/$leg $N --n 512 \
         && python3 perf/rfd4/e2e/census_v2.py $R/$leg $R/e2e-512-main.json) > $R/$leg.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
$BOOK rm $OWN > /dev/null
stamp ALLDONE
