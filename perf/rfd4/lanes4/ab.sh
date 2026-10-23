#!/bin/bash
# rfd4-perf-lanes4 paired A/B on one qb2 chip: A = origin/main, B = branch (next lane launched before the heads).
# Two ABAB rounds of 32 designs a size at 256 w4 and 128 w16 (perf/rfd4/b128/designs.py), then one split.py a side at 256.
#   bash ab.sh <chip> <outdir> <sha_A> <sha_B>
set -u
N=$1; R=$2; A=$3; B=$4
WT=/home/ttuser/.coworker/wt/rfd4-perf-lanes4
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
IN=$HOME/rfd4-inputs; W=$HOME/rfd4-bundle
mkdir -p $R; R=$(cd $R && pwd)
for arm in A B; do sha=${!arm}; t=$R/tree-$arm; mkdir -p $t; git -C $WT archive $sha | tar -x -C $t; echo $sha > $t/TREE; done
cp $WT/perf/rfd4/lanes4/split.py $R/tree-A/perf/rfd4/lanes4/ 2>/dev/null || { mkdir -p $R/tree-A/perf/rfd4/lanes4; cp $WT/perf/rfd4/lanes4/split.py $R/tree-A/perf/rfd4/lanes4/; }
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-lanes4
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
leg() { arm=$1 n=$2 w=$3 seed=$4; t=$R/tree-$arm; o=$R/r$5-$arm-n$n
  (cd $t && PYTHONPATH=$t:$t/perf timeout 1800 $PY perf/rfd4/b128/designs.py --spec $IN/pdl1-$n.yaml --weights $W \
     --out $o --width $w --designs 32 --seed $seed) > $o.log 2>&1
  stamp "r$5-$arm-n$n rc=$?"; }
stamp "start chip $N A=$A B=$B"
for r in 1 2; do s=$(( (r-1)*100 ))
  leg A 256 4 $s $r; leg B 256 4 $s $r; leg A 128 16 $s $r; leg B 128 16 $s $r
done
for arm in A B; do t=$R/tree-$arm
  (cd $t && PYTHONPATH=$t:$t/perf timeout 1800 $PY perf/rfd4/lanes4/split.py --spec $IN/pdl1-256.yaml --weights $W \
     --out $R/split-$arm-n256 --width 4 --designs 32) > $R/split-$arm-n256.log 2>&1
  stamp "split-$arm rc=$?"
done
stamp ALLDONE
