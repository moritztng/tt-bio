#!/bin/bash
# Run one short z8 device leg on the first qb2 card whose lock frees (release gates keep priority; up to 4 h).
#   perf/rfd4/z8/probe_any.sh <out dir> <script> [args...]     (run from the tree)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=${1:?out}; shift; PY=$HOME/tt-bio-dev/env/bin/python
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card for $*"
end=$(( $(date +%s) + 14400 ))
while [ $(date +%s) -lt $end ]; do
  for C in 1 2 0 3; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $C --except rfd4-perf-z8 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$C.lock"
    if flock -n 9; then
      fuser "/dev/tenstorrent/$C" >/dev/null 2>&1 && { exec 9>&-; continue; }
      python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 900 --now --note "z8 short leg" >/dev/null
      stamp "card $C"
      TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8 PYTHONPATH=$W \
        timeout 600 $PY "$@" >> $O/leg.log 2>&1
      stamp "rc=$? card $C"
      exec 9>&-
      python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
      stamp ALLDONE; exit 0
    fi
    exec 9>&-
  done
  sleep 30
done
stamp TIMEOUT
