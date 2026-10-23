#!/bin/bash
# t1k run1: where T1's time goes at RFD4's key (c_z 384, hidden 256), the default one-launch plan (bfp8 resident
# weights, HiFi2, packer sigmoid). perf/spd_trimul/in_t1.py times the real kernel, then each ABL stage-skip mask
# (kernels/trimul_in_t1: 1 LN, 2 g_out mm, 4 [a|b] mm, 8 gate, 16 z reads, 32 gather, 64 writes, 512 sigmoid), so
# each stage's cost reads by difference; AICLK sampled during every N. Then the writer knobs at 512 and the mask case.
#   perf/rfd4/t1k/run1.sh <card> [legs]     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/perf/rfd4/t1k/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-abl knobs mask}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1k PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1k ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
T1="--c 384 --hidden 256 --w8 --fid HiFi2"
ABL=16,32,64,96,112,1,2,4,6,8,512,527,639
for leg in $LEGS; do
  case $leg in
    abl) timeout 2400 $PY perf/spd_trimul/in_t1.py $T1 --no-mask --ns 512,384,256 --set T1_PSIG=1 --abl $ABL \
           --out $O/abl.jsonl > $O/abl.log 2>&1 ;;
    knobs) for k in GIN_MOVE_NB=4 GIN_MOVE_NB=8 GIN_MOVE_NB=1; do
             timeout 900 $PY perf/spd_trimul/in_t1.py $T1 --no-mask --ns 512 --set T1_PSIG=1,$k \
               --out $O/knobs.jsonl >> $O/knobs.log 2>&1; done ;;
    mask) timeout 900 $PY perf/spd_trimul/in_t1.py $T1 --ns 512 --set T1_PSIG=1 --abl 112 \
            --out $O/mask.jsonl > $O/mask.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1k
stamp ALLDONE
