#!/bin/bash
# z8 run6: the four z-graded trunk tests on main's tree ($OLD) for their device readings beside run5's (pair_bfp8z
# default). Waits for run5's ALLDONE, then the first free qb2 card.   perf/rfd4/z8/run6.sh <main tree>
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/z8/run6; PY=$HOME/tt-bio-dev/env/bin/python; OLD=${1:?main tree}
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
until grep -q ALLDONE $W/artifacts/z8/run5/run.log 2>/dev/null; do sleep 30; done
C=
while [ -z "$C" ]; do
  for c in 3 1 2 0; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $c --except rfd4-perf-z8 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    flock -n 9 || { exec 9>&-; continue; }
    fuser "/dev/tenstorrent/$c" >/dev/null 2>&1 && { exec 9>&-; continue; }
    C=$c; break
  done
  [ -z "$C" ] && sleep 30
done
python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 1800 --now --note "z8 main-tree trunk tests" >/dev/null
stamp "card $C main $(cat $OLD/TREE_SHA)"
(cd $OLD && TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8 RFD4_GOLDENS=$HOME/rfd4-goldens \
  PYTHONPATH=$OLD timeout 1500 $PY -m pytest -m device -q -s tests/test_rfd4_trunk_golden.py::test_released_trunk_14x6_recycles \
  tests/test_rfd4_trunk_parity.py::test_trunk_stack tests/test_rfd4_trunk_parity.py::test_trunk_run_bucketed_ragged_length) > $O/main.log 2>&1
stamp "main rc=$?"
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
stamp ALLDONE
