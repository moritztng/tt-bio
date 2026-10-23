#!/bin/bash
# tri6 run2: T1 + T2 weight multicast together (trimul_tail.T1_MCAST, T2_MCAST) in the paired traced trunk.
#   ab512 ab256    base arm with both off, 30 reps
#   t2ab512 t2ab256  base arm with T2_MCAST off only (T1 on both arms): the tail's share
#   perf/rfd4/tri6/run2.sh <card> [after-log] [legs]   (run from the tree; waits for after-log's ALLDONE, books the card)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/tri6/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; AFTER=${2:-}; LEGS=${3:-ab512 ab256 t2ab512 t2ab256}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri6
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
[ -n "$AFTER" ] && { stamp "wait $AFTER"; until grep -q ALLDONE "$AFTER" 2>/dev/null; do sleep 30; done; }
python3 $HOME/spd/feed/book.py add rfd4-perf-tri6 $C 3600 --note "RFD4 bet 33 tri6 run2: T1+T2 weight mcast trunk A/B; contact rfd4-orchestrator" | tee -a $O/run.log
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-tri6 ]; do
  python3 $HOME/spd/feed/book.py add rfd4-perf-tri6 $C 3600 >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$(PYTHONPATH=$W $PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in
    ab*) OFF="trimul_tail.T1_MCAST={};trimul_tail.T2_MCAST=set()"; n=${leg#ab} ;;
    t2ab*) OFF="trimul_tail.T2_MCAST=set()"; n=${leg#t2ab} ;;
  esac
  PYTHONPATH=$W:$W/perf TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py \
    --weights $HOME/rfd4-bundle --out $O/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "$OFF" \
    > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-tri6
stamp ALLDONE
