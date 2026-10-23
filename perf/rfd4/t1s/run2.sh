#!/bin/bash
# t1s run2: which data-movement stage binds T1 once both cores gather. run1 read compute-only 2.73 ms and DM-only
# 2.68 at 512 with the split (main: 2.80 / 3.03), so the split took 0.35 ms off the DM side, not the ~1.2 a halved
# writer would. Each DM stage skipped with compute on and with compute skipped (527 = LN, both matmuls, gate,
# sigmoid), then the new default and main's T1 alternated three times (qb2's load moves a single reading ~0.2 ms).
#   perf/rfd4/t1s/run2.sh <card> [legs]     (run from the tree; starts after run1's ALLDONE, books the card itself)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1s/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-abl rep}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait run1"
until grep -q ALLDONE $W/artifacts/t1s/run1/run.log 2>/dev/null; do sleep 20; done
python3 $HOME/spd/feed/book.py add rfd4-perf-t1s $C 1800 >/dev/null 2>&1
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2 --no-mask"
t1() { local leg=$1; shift; timeout 1500 $PY perf/spd_trimul/in_t1.py $R "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
for leg in $LEGS; do
  case $leg in
    abl) t1 abl --ns 512,256 --set T1_PSIG=1 --abl 16,32,64,96,112,527,543,559,591,623,639 --reps 20 ;;
    rep) for i in 1 2 3; do
           t1 rep --ns 512,384,256 --set T1_PSIG=1 --reps 20
           t1 rep --ns 512,384,256 --set T1_PSIG=1,T1_SPLIT=0,T1_LNB=1 --reps 20
         done ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
