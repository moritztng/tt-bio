#!/bin/bash
# k256 run5: `token_mm2d` in the trunk, paired traced A/B on RFD4's default set, 30 reps at 256 and 512, AICLK
# sampled during. 512 must read z bit-identical (the key is ttnn's own plan there) and not regress.
#   perf/rfd4/k256/run5.sh <card> [legs]     (run from the tree; books the card and waits for its window)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/k256/${RUN:-run5}; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-ab256 ab512}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-k256 PYTHONPATH=$W:$W/perf
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "book chip$C $(python3 $HOME/spd/feed/book.py add rfd4-perf-k256 $C 1500 --note 'RFD4 bet 31 k256 run5 token_mm2d')"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-k256 ]; do
  sleep 30; python3 $HOME/spd/feed/book.py add rfd4-perf-k256 $C 1500 --note 'RFD4 bet 31 k256 run5 token_mm2d' >/dev/null
done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
B=$($PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  n=${leg#ab}
  TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py --weights $HOME/rfd4-bundle --out $O/$leg.json \
    --n $n --reps 30 --base $B --levers $B,token_mm2d > $O/$leg.log 2>&1
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-k256
stamp ALLDONE
