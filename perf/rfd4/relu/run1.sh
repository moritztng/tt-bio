#!/bin/bash
# relu run1: pair_relu per fidelity standalone (256, 512), then the paired traced trunk, RFD4 default vs default +
# pair_block_hifi2, 30 reps at 512 and 256. Books the card on qb2's arbiter and waits for it.
#   perf/rfd4/relu/run1.sh <card> [legs]     (run from the tree on qb2)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/relu/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-solo ab512 ab256}; OWN=rfd4-perf-relu
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:$OWN PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "book"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = $OWN ]; do
  python3 $HOME/spd/feed/book.py add $OWN $C 2700 --note "RFD4 bet 32 relu run1" >/dev/null 2>&1; sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  if [ $leg = solo ]; then
    timeout 1200 $PY perf/rfd4/pairblock/transition.py --weights $HOME/rfd4-bundle --out $O/solo.json --n 256 512 \
      --arms fused_hifi4 fused_hifi2 fused_lofi --reps 7 --inner 10 > $O/solo.log 2>&1
  else
    n=${leg#ab}
    TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle \
      --out $O/$leg.json --n $n --reps 30 --base $B --levers $B,pair_block_hifi2 > $O/$leg.log 2>&1
  fi
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm $OWN
stamp ALLDONE
