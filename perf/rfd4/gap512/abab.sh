#!/bin/bash
# rfd4-perf-gap512 paired ABAB after the split chain: A (main + tools) vs P (two calls in flight), width-1 lanes,
# 8 designs a leg, two rounds = 16 designs a side, at 512 then 384, one chip one session (run.sh books and flocks).
# Then the card-free suite on both trees, card masked, after the timed legs so it does not load the host under them.
#   A=<sha> P=<sha> bash abab.sh <chip> <outdir> [wait_for_dir]
set -u
N=$1; R=$2; WAIT=${3:-}; HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $R; R=$(cd $R && pwd)
[ -n "$WAIT" ] && until grep -q ALLDONE $WAIT/done 2>/dev/null; do sleep 30; done
DESIGNS=8 SIZE=512 HOLD_S=3600 bash $HERE/run.sh $N $R/n512 "sA1 sP1 sA2 sP2"
DESIGNS=8 SIZE=384 HOLD_S=2400 bash $HERE/run.sh $N $R/n384 "sA1 sP1 sA2 sP2"
PY=$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python
for arm in A P; do t=$R/n512/tree-$arm
  (cd $t && TT_VISIBLE_DEVICES=99 PYTHONPATH=$t timeout 3000 $PY -m pytest -q -p no:cacheprovider -m "not device" \
     -k "rfd4 or trimul or precision_levers or protenix" tests) > $R/suite-$arm.log 2>&1
  echo "suite-$arm rc=$? $(date -u +%FT%TZ)" >> $R/done
done
echo "ALLDONE $(date -u +%FT%TZ)" >> $R/done
