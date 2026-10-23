#!/bin/bash
# z8 run7: perf/rfd4/z8/diag.py (padded vs full, 1 vs 6 cycles, bf16 vs bfp8 z, device vs the float64 storage model)
# on the first free qb2 card.   RUN=run7 perf/rfd4/z8/run7.sh [diag args]   (run from the tree)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/z8/${RUN:-run7}; PY=$HOME/tt-bio-dev/env/bin/python
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card"
C=; end=$(( $(date +%s) + 14400 ))
while [ -z "$C" ] && [ $(date +%s) -lt $end ]; do
  for c in 3 1 2 0; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $c --except rfd4-perf-z8 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    flock -n 9 || { exec 9>&-; continue; }
    fuser "/dev/tenstorrent/$c" >/dev/null 2>&1 && { exec 9>&-; continue; }
    C=$c; break
  done
  [ -z "$C" ] && sleep 30
done
[ -z "$C" ] && { stamp TIMEOUT; exit 3; }
python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 3600 --now --note "z8 diag" >/dev/null
stamp "card $C tree $(cat TREE_SHA 2>/dev/null)"
TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8 PYTHONPATH=$W OMP_NUM_THREADS=8 \
  timeout 2700 $PY perf/rfd4/z8/diag.py --out $O/diag.jsonl "$@" > $O/diag.log 2>&1
stamp "diag rc=$?"
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
stamp ALLDONE
