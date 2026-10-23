#!/bin/bash
# t1s run1: T1 at RFD4's key with both data-movement cores gathering (trimul_tail.T1_SPLIT) and the layer_norm in
# 4-tile DST blocks (T1_LNB), against main (t1k's NB 8, writer alone, LN BLK 1).
#   t1    the new default, 512/384/256: each lever turned off alone (--ab, a/b/g_out torch.equal), DM-only / compute-only
#   main  main's T1 (both off) with each lever turned on alone
#   mask  the padded case at 512
#   xtree a/b/g_out digests at every model key on this tree and on main's tree ($OLD): equal = bit-identical
#   ab512 ab256 ab384  paired traced trunk, base arm built with both levers off (stack_ab --base-set), 30 reps
#   perf/rfd4/t1s/run1.sh <card> <main tree> [legs]     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/t1s/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; OLD=${2:?main tree}; LEGS=${3:-t1 main mask xtree ab512 ab256 ab384}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-t1s
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-t1s ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null) main $(cat $OLD/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2"
t1() { local tree=$1 leg=$2; shift 2
  PYTHONPATH=$tree timeout 1500 $PY perf/spd_trimul/in_t1.py "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
OFF="trimul_tail.T1_SPLIT={};trimul_tail.T1_LNB={}"
B=$(PYTHONPATH=$W $PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in
    t1) t1 $W $leg $R --no-mask --ns 512,384,256 --set T1_PSIG=1 --ab T1_SPLIT=0,T1_LNB=1 --abl 112,527,96 ;;
    main) t1 $W $leg $R --no-mask --ns 512,384,256 --set T1_PSIG=1,T1_SPLIT=0,T1_LNB=1 --ab T1_SPLIT=1,T1_LNB=4 \
            --abl 112,527 ;;
    mask) t1 $W $leg $R --ns 512 --set T1_PSIG=1 --ab T1_SPLIT=0,T1_LNB=1 ;;
    xtree) for tree in $W $OLD; do
             t1 $tree xtree $R --no-mask --ns 512,256 --set T1_PSIG=1 --reps 3
             t1 $tree xtree $R --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 384 --hidden 384 --no-mask --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 128 --hidden 128 --no-mask --ns 512 --reps 3
           done ;;
    ab*) n=${leg#ab}
      PYTHONPATH=$W:$W/perf TT_BIO_TRACE_REGION_MB=512 timeout 2400 $PY perf/rfd4/bfp8/stack_ab.py \
        --weights $HOME/rfd4-bundle --out $O/$leg.json --n $n --reps 30 --base $B --levers $B --base-set "$OFF" \
        > $O/$leg.log 2>&1 ;;
  esac
  stamp "$leg rc=$?"
done
exec 9>&-
python3 $HOME/spd/feed/book.py rm rfd4-perf-t1s
stamp ALLDONE
