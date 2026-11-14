#!/bin/bash
# tri6 run1: T1 weight multicast (trimul_tail.T1_MCAST, RFD4's key) against every core reading its own copy.
#   t1     standalone T1 at RFD4's default (w8 HiFi2 psig b8), 512/256: T1_MCAST=0 via --ab (a/b/g_out torch.equal + ms),
#          and the all-stage skip (127) both ways: what the weight load costs
#   xtree  a/b/g_out digests at every model key on this tree and on main's tree ($OLD): equal = other models untouched
#   ab512 ab256  paired traced trunk, base arm with T1_MCAST={} (stack_ab --base-set), 30 reps, AICLK during
#   perf/rfd4/tri6/run1.sh <card> <main tree> [legs]     (run from the tree; waits for the card's booking)
set -u
W=$(cd "$(dirname "$0")/../../.." && pwd); O=$W/artifacts/tri6/run1; PY=$HOME/tt-bio-dev/env/bin/python
C=${1:?card}; OLD=${2:?main tree}; LEGS=${3:-t1 xtree ab512 ab256}
export TT_VISIBLE_DEVICES=$C TT_BIO_LEASE_CARDS=$C TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri6
mkdir -p $O; cd $W
stamp() { echo "$1 $(date -u +%FT%TZ) load $(cut -d" " -f1 /proc/loadavg)" | tee -a $O/run.log; }
stamp "wait booking chip$C"
until [ "$(python3 $HOME/spd/feed/book.py active $C 2>/dev/null)" = rfd4-perf-tri6 ]; do sleep 30; done
exec 9>"$HOME/spd_qb2_chip$C.lock"
flock -w 3600 9 || { stamp "flock timeout"; exit 3; }
while fuser "/dev/tenstorrent/$C" >/dev/null 2>&1; do sleep 20; done
stamp "start tree $(cat $W/TREE_SHA 2>/dev/null) main $(cat $OLD/TREE_SHA 2>/dev/null)"
R="--c 384 --hidden 256 --w8 --fid HiFi2"
t1() { local tree=$1 leg=$2; shift 2
  PYTHONPATH=$tree timeout 1500 $PY perf/spd_trimul/in_t1.py "$@" --out $O/$leg.jsonl >> $O/$leg.log 2>&1; }
OFF="trimul_tail.T1_MCAST={}"
B=$(PYTHONPATH=$W $PY -c "from tt_bio.rfd4.trunk import NORMAL_LEVERS as N; print(','.join(sorted(N)))")
for leg in $LEGS; do
  case $leg in
    t1) t1 $W $leg $R --no-mask --ns 512,256,512,256 --reps 20 --set T1_PSIG=1,T1_B8=1 --ab T1_MCAST=0 --abl 127
        t1 $W $leg $R --no-mask --ns 512,256 --reps 20 --set T1_PSIG=1,T1_B8=1,T1_MCAST=0 --ab T1_MCAST=1 --abl 127 ;;
    xtree) for tree in $W $OLD; do
             t1 $tree xtree $R --no-mask --ns 512,256 --set T1_PSIG=1,T1_B8=1 --reps 3
             t1 $tree xtree $R --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 256 --hidden 256 --w8 --fid HiFi3 --no-mask --ns 512 --set T1_PSIG=1 --reps 3
             t1 $tree xtree --c 384 --hidden 384 --w8 --no-mask --ns 512 --set T1_PSIG=1 --reps 3
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
python3 $HOME/spd/feed/book.py rm rfd4-perf-tri6
stamp ALLDONE
