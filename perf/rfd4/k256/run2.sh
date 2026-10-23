#!/bin/bash
# k256 run2: the triangle contraction alone at 256 and 512 per plan (trunk mm config, ttnn plain, bmm_res, HiFi2):
# where 256 sits against 1/8 of 512 per call. perf/rfd4/tri3/contract.py, arms interleaved, AICLK sampled.
#   perf/rfd4/k256/run2.sh <card>     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/k256/run2; PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
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
for n in 256 512; do
  timeout 900 $PY perf/rfd4/tri3/contract.py --n $n --rounds 4 --reps 8 --out $O/contract-$n.json \
    --arms "mm|mm;fid=hifi2|plain|bres|bres;fid=hifi2" > $O/contract-$n.log 2>&1
  stamp "contract $n rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-k256
stamp ALLDONE
