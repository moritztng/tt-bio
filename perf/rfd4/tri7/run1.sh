#!/bin/bash
# tri7 run1: the tail's two levers, G8 (trimul_t1g8, g_out bfp8) and H (trimul_tail_hifi2, the p GEMM at HiFi2).
#   tail   standalone RFD4 tail (perf/rfd4/tri7/tail.py) 512/384/256: base / h / g8 / hg8 ms, error vs float64, the
#          T2_ABL stage split of the base (2 GEMM, 8 epilogue, 11 all compute but sigmoid, 27 all compute)
#   t1     standalone T1 at RFD4's default, T1_G8 on vs off (--ab): ms and g_out error
#   xtree  each tree's own scripts: tail digests at RFD4's and other models' T2 keys, T1 digests at every key,
#          this tree vs origin/main ($OLD): equal = other models untouched
#   ab<lever><n>  paired traced trunk, 30 reps, base = NORMAL_LEVERS, stack = + lever(s); lever g8 / h / hg8
#   perf/rfd4/tri7/run1.sh <card> <main tree> [legs]     (run from a snapshot tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/tri7/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; OLD=${2:?main tree}
LEGS=${3:-tail t1 xtree abg8512 abh512 abhg8512 abg8256 abh256 abhg8256 abg8384 abh384 abhg8384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri7
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-tri7 ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null) main $(cat $OLD/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2"
B=$($PY -c "import sys; sys.path.insert(0, '$W'); from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in
    tail) timeout 1500 $PY perf/rfd4/tri7/tail.py --ns 512,384,256 --abl 2,8,11,27 --out $O/tail.jsonl >> $O/tail.log 2>&1 ;;
    t1) timeout 1500 $PY perf/spd_trimul/in_t1.py $R --no-mask --ns 512,256,512,256 --reps 20 \
          --set T1_PSIG=1,T1_B8=1 --ab T1_G8=1 --out $O/t1.jsonl >> $O/t1.log 2>&1 ;;
    xtree) for tree in $W $OLD; do
             (cd $tree
              timeout 900 $PY perf/rfd4/tri7/tail.py --ns 512,256 --variants base --reps 3 --xtree --out $O/xtree_tail.jsonl
              t1() { timeout 900 $PY perf/spd_trimul/in_t1.py "$@" --reps 3 --out $O/xtree_t1.jsonl; }
              t1 $R --no-mask --ns 512,256 --set T1_PSIG=1,T1_B8=1
              t1 $R --ns 512 --set T1_PSIG=1
              t1 --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512 --set T1_PSIG=1
              t1 --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1
              t1 --c 128 --hidden 128 --no-mask --ns 512) >> $O/xtree.log 2>&1
           done ;;
    ab*) spec=${leg#ab}; n=${spec: -3}; lv=${spec%???}
      case $lv in g8) X=trimul_t1g8 ;; h) X=trimul_tail_hifi2 ;; hg8) X=trimul_t1g8,trimul_tail_hifi2 ;; esac
      PYTHONPATH=$W:$W/perf TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py \
        --weights $HOME/rfd4-bundle --out $O/$leg.json --n $n --reps 30 --base $B --levers $B,$X \
        > $O/$leg.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-tri7
stamp ALLDONE
