#!/bin/bash
# k256 run4: the token-track linears alone at 256 and 512 per program config (production, plain, 1D, 2D):
# perf/rfd4/k256/tok_mm.py, arms interleaved, AICLK sampled.
#   perf/rfd4/k256/run4.sh <card>     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/k256/run4; PY=${PY:-$HOME/tt-bio-dev/env/bin/python}
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
  timeout 900 $PY perf/rfd4/k256/tok_mm.py --n $n --out $O/tok-$n.json > $O/tok-$n.log 2>&1
  stamp "tok $n rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-k256
stamp ALLDONE
