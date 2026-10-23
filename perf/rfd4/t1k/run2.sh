#!/bin/bash
# t1k run2: T1's writer gathers more tiles per read barrier (trimul_tail.T1_NB). run1 read NB 2 -> 4 as 3.59 -> 3.30 ms
# at 512 (AICLK 1350): the writer, not the FPU, sets T1's time. Per model key: each NB alone against the default,
# a / b / g_out bit-equal and ms (in_t1.py --ab), RFD4 also with compute-only / DM-only times at NB 8.
#   perf/rfd4/t1k/run2.sh <card> [legs]     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1k/run2; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; LEGS=${2:-rfd4 rfd4nb8 rfd4mask protenix opendde opendde8 boltz}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1k PYTHONPATH=$W
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1k ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null)"
AB=T1_NB=2,T1_NB=4,T1_NB=8,T1_NB=16
R="--c 384 --hidden 256 --w8 --fid HiFi2"
run() { local leg=$1; shift; timeout 1500 $PY perf/spd_trimul/in_t1.py "$@" --out $O/$leg.jsonl > $O/$leg.log 2>&1; }
for leg in $LEGS; do
  case $leg in
    rfd4) run $leg $R --no-mask --ns 512,384,256 --set T1_PSIG=1 --ab $AB --abl 112,527 ;;
    rfd4nb8) run $leg $R --no-mask --ns 512,384,256 --set T1_PSIG=1,T1_NB=8 --abl 16,32,64,112,1,6,527 ;;
    rfd4mask) run $leg $R --ns 512 --set T1_PSIG=1 --ab T1_NB=4,T1_NB=8 ;;
    protenix) run $leg --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512,256 --set T1_PSIG=1 --ab $AB ;;
    opendde) run $leg --c 384 --hidden 384 --no-mask --ns 512 --set T1_PSIG=1 --ab $AB ;;
    opendde8) run $leg --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1 --ab $AB ;;
    boltz) run $leg --c 128 --hidden 128 --no-mask --ns 512,256 --ab $AB ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1k
stamp ALLDONE
