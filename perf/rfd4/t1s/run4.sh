#!/bin/bash
# t1s run4: lever A alone in the trunk (T1_SPLIT). run3 read B, the LN blocks, at 0.998x paired at 512, so B was
# dropped; this tree has A only. Paired traced A/B, base arm built with T1_SPLIT {} (main), 30 reps, z bit-identical.
#   perf/rfd4/t1s/run4.sh <card> <run3 log> [legs]     (run from the tree; starts after run3's ALLDONE, books the card)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1s/run4; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; R3=${2:?run3 log}; LEGS=${3:-ab512 ab256 ab384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait run3"
until grep -q ALLDONE $R3 2>/dev/null; do sleep 20; done
python3 $HOME/spd/feed/book.py add rfd4-perf-t1s $C 1200 >/dev/null 2>&1
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$leg.json \
    --n $n --reps 30 --base $B --levers $B --base-set "trimul_tail.T1_SPLIT={}" > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
