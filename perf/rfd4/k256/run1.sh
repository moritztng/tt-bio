#!/bin/bash
# k256 run1: T1 at RFD4's key with 4-row work items (eighths of a unit) against production's 8-row items, at 256
# (78 -> 93 % round fill on 110 cores) and 512 (93 % either way, the control). in_t1.py --ab checks a, b, g_out bit
# equality against the default and times both; AICLK is sampled during every N.
#   perf/rfd4/k256/run1.sh <card>     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/k256/${RUN:-run1}; PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
C=${1:?card}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-k256 PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-k256 ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
timeout 1500 $PY perf/spd_trimul/in_t1.py --c 384 --hidden 256 --w8 --fid HiFi2 --no-mask --ns 256,512,256,512 \
  --set T1_PSIG=1 --ab T1_ROWS=4 --out $O/ab.jsonl > $O/ab.log 2>&1
stamp "ab rc=$?"
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-k256
stamp ALLDONE
