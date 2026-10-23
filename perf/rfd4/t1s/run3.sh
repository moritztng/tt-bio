#!/bin/bash
# t1s run3: each lever's own share in the trunk. run1's standalone readings of B (T1_LNB) alone disagree in sign
# across legs (+-0.1-0.2 ms at this host load), so the paired traced trunk prices it: base arm built with T1_LNB {}
# (split only) and, as a check, with T1_SPLIT {} (LN blocks only). 30 reps, z must stay bit-identical.
#   perf/rfd4/t1s/run3.sh <card> [legs]     (run from the tree; starts after run2's ALLDONE, books the card itself)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1s/run3; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-lnb512 split512 lnb256 lnb384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait run2"
until grep -q ALLDONE $W/artifacts/t1s/run2/run.log 2>/dev/null; do sleep 20; done
python3 $HOME/spd/feed/book.py add rfd4-perf-t1s $C 1200 >/dev/null 2>&1
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in lnb*) n=${leg#lnb}; S="trimul_tail.T1_LNB={}" ;; split*) n=${leg#split}; S="trimul_tail.T1_SPLIT={}" ;; esac
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$leg.json \
    --n $n --reps 30 --base $B --levers $B --base-set "$S" > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
