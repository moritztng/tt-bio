#!/bin/bash
# t1s run6: lever C with t1_pair_free (bfp8 a / b only where pairing keeps the busiest core's item count). Paired traced
# trunk, base = RFD4 default, stack = + trimul_t1b8, 30 reps: 256 must read bit-identical now (p300c keeps bf16 there),
# 512 re-read on this tree.
#   perf/rfd4/t1s/run6.sh <card> <run5 log> [legs]     (run from the tree; starts after run5's ALLDONE, books the card)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1s/run6; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; R5=${2:?run5 log}; LEGS=${3:-ab256 ab512 ab384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait run5"
until grep -q ALLDONE $R5 2>/dev/null; do sleep 20; done
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do
  python3 $HOME/spd/feed/book.py add rfd4-perf-t1s $C 900 --note "RFD4 bet 27 t1s run6" >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$leg.json \
    --n $n --reps 30 --base $B --levers $B,trimul_t1b8 > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
