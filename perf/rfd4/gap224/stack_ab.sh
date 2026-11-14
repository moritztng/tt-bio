#!/bin/bash
# One trunk call timed on the ttnn wheel (W) and on the Tracy source build with the profiler off (S), interleaved,
# at the given sizes: does the census (S with the profiler on) disagree with the timed call because of the build?
#   bash perf/rfd4/gap224/stack_ab.sh <chip> <outdir> "<n>-b<w> ..." [rounds]
set -u
N=$1; R=$2; SIZES=$3; ROUNDS=${4:-2}; WT=$(cd "$(dirname "$0")/../../.." && pwd)
WPY=${WPY:-$HOME/scratch/shipflow/venv312-1384f4c1b7bf/bin/python}
M=$HOME/tt-metal-b2z; BOOK="python3 $HOME/spd/feed/book.py"; OWN=rfd4-perf-gap224
mkdir -p $R; R=$(cd $R && pwd); cd $WT
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:$OWN TT_BIO_TRACE_REGION_MB=2048
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" >> $R/done; }
until [ "$($BOOK active $N 2>/dev/null)" = $OWN ]; do $BOOK add $OWN $N ${HOLD_S:-2400} > /dev/null; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$N.lock"; flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$N" > /dev/null 2>&1; do sleep 15; done
stamp "start chip $N tree $(git rev-parse --short HEAD)"
for r in $(seq 1 $ROUNDS); do
  for sz in $SIZES; do
    n=${sz%-b*}; w=${sz##*-b}
    for arm in W S; do
      leg=$arm-$sz@$r
      if [ $arm = W ]; then
        PYTHONPATH=$WT:$WT/perf timeout 900 $WPY perf/rfd4/gap224/call_bench.py --n $n --batch $w --out $R/$leg.json
      else
        (unset LD_LIBRARY_PATH; TT_METAL_HOME=$M PYTHONPATH=$WT:$WT/perf:$M/ttnn:$M/tools:$M \
          timeout 900 /home/ttuser/tt-bio-dev/env/bin/python perf/rfd4/gap224/call_bench.py --n $n --batch $w --out $R/$leg.json)
      fi > $R/$leg.log 2>&1
      stamp "$leg rc=$?"
    done
  done
done
exec 9>&-; $BOOK rm $OWN; stamp ALLDONE
