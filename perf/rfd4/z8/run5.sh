#!/bin/bash
# z8 run5, on the tree with pair_bfp8z in RFD4's default set: the RFD4 / trimul device tests (as LAND12 ran them), and
# T1 / T2 digests at every model's key on this tree and on main's ($OLD): equal = other models untouched.
#   RUN=run5 perf/rfd4/z8/run5.sh <main tree> [legs]     (run from the tree; first free qb2 card, up to 4 h)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/z8/${RUN:-run5}; PY=$HOME/tt-bio-dev/env/bin/python
OLD=${1:?main tree}; LEGS=${2:-devtests xtree}
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait any card, legs $LEGS"
end=$(( $(date +%s) + 14400 )); C=
while [ -z "$C" ] && [ $(date +%s) -lt $end ]; do
  for c in 1 2 0 3; do
    [ -n "$(python3 $HOME/spd/feed/book.py active $c --except rfd4-perf-z8 2>/dev/null)" ] && continue
    exec 9>"$HOME/spd_qb2_chip$c.lock"
    flock -n 9 || { exec 9>&-; continue; }
    fuser "/dev/tenstorrent/$c" >/dev/null 2>&1 && { exec 9>&-; continue; }
    C=$c; break
  done
  [ -z "$C" ] && sleep 30
done
[ -z "$C" ] && { stamp TIMEOUT; exit 3; }
python3 $HOME/spd/feed/book.py add rfd4-perf-z8 $C 3600 --now --note "z8 device tests + xtree" >/dev/null
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-z8
stamp "card $C start $(cat TREE_SHA 2>/dev/null) main $(cat $OLD/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2"
for leg in $LEGS; do
  case $leg in
    devtests) RFD4_GOLDENS=$HOME/rfd4-goldens PYTHONPATH=$W timeout 1800 $PY -m pytest -m device -q -s \
                tests/test_rfd4_*.py tests/test_trimul_*.py > $O/devtests.log 2>&1 ;;
    xtree) for tree in $W $OLD; do
             (cd $tree; export PYTHONPATH=$tree
              echo "tree $tree" >> $O/xtree_tail.jsonl; echo "tree $tree" >> $O/xtree_t1.jsonl
              timeout 900 $PY perf/rfd4/tri7/tail.py --ns 512,256 --variants base --reps 3 --xtree --out $O/xtree_tail.jsonl
              t1() { timeout 900 $PY perf/spd_trimul/in_t1.py "$@" --reps 3 --out $O/xtree_t1.jsonl; }
              t1 $R --no-mask --ns 512,256 --set T1_PSIG=1,T1_B8=1
              t1 $R --ns 512 --set T1_PSIG=1
              t1 --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512 --set T1_PSIG=1
              t1 --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1
              t1 --c 128 --hidden 128 --no-mask --ns 512) >> $O/xtree.log 2>&1
           done ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-z8
stamp ALLDONE
