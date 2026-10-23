#!/bin/bash
# t1k run3: T1_NB 8 at RFD4's key in the trunk. Paired traced A/B (perf/rfd4/bfp8/stack_ab.py, 30 reps, AICLK sampled),
# both arms RFD4's default lever set; the base arm builds its programs with trimul_tail.T1_NB = {} (gin_moved's NB 2,
# main's behaviour). z after one pass must be bit-identical (z_rel_rms_stack_vs_base 0).
#   perf/rfd4/t1k/run3.sh <card> [legs]     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1k/run3; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-ab512 ab256 ab384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1k PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1k ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$leg.json \
    --n $n --reps 30 --base $B --levers $B --base-set "trimul_tail.T1_NB={}" > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1k
stamp ALLDONE
